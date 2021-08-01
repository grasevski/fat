//! Deep reinforcement learning agent.
use super::{cfg, fxcm};
use arrayvec::ArrayVec;
use rand::{prelude::SliceRandom, rngs::StdRng, SeedableRng};
use std::convert::{TryFrom, TryInto};
use tch::{
    manual_seed, nn,
    nn::{Module, OptimizerConfig, RNN},
    no_grad, Device, Kind, Reduction, TchError, Tensor,
};

/// Array type for big hidden batch on the stack.
pub type HiddenBatch = ArrayVec<f32, { cfg::BATCH * fxcm::LAYER_DIM * cfg::FEATURES }>;

/// Array type for big observation batch on the stack.
pub type ObservationBatch = ArrayVec<f32, { cfg::BATCH * cfg::WINDOW * 2 * fxcm::Symbol::N }>;

/// Gradient descent algorithm.
type Optimizer = nn::AdamW;

/// Reinforcement learning agent.
pub struct Model {
    /// Number of epochs per update.
    epochs: u8,

    /// Random number generator, for shuffling batches.
    rng: StdRng,

    /// PyTorch heap, may be on either GPU or CPU.
    vs: nn::VarStore,

    /// Gradient descent method.
    opt: nn::Optimizer<Optimizer>,

    /// Stateful time series decoder.
    rnn: nn::GRU,

    /// Which position to hold for each symbol.
    actor: nn::Linear,

    /// Value of the current state.
    critic: nn::Linear,
}

impl Model {
    /// Initializes the neural network.
    pub fn new(
        epochs: u8,
        seed: i64,
        output: u8,
        dropout: f64,
        learning_rate: f64,
        has_biases: bool,
    ) -> fxcm::Result<Self> {
        let rng = StdRng::seed_from_u64(seed.try_into()?);
        manual_seed(seed);
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let opt = Optimizer::default().build(&vs, learning_rate)?;
        let cfg = nn::RNNConfig {
            has_biases,
            num_layers: cfg::LAYERS.try_into()?,
            dropout,
            bidirectional: cfg::BIDIRECTIONAL,
            ..Default::default()
        };
        let rnn = nn::gru(
            &(&vs.root() / "rnn"),
            (2 * fxcm::Symbol::N).try_into()?,
            cfg::FEATURES.try_into()?,
            cfg,
        );
        let actor = Self::predictor(&vs.root() / "actor", output)?;
        let critic = Self::predictor(&vs.root() / "critic", output)?;
        Ok(Self {
            epochs,
            rng,
            vs,
            opt,
            rnn,
            actor,
            critic,
        })
    }

    /// Constructs a prediction head.
    fn predictor(path: nn::Path, output: u8) -> fxcm::Result<nn::Linear> {
        let mut input = cfg::WINDOW * cfg::FEATURES;
        if cfg::BIDIRECTIONAL {
            input <<= 1;
        }
        input += usize::from(output);
        Ok(nn::linear(
            path,
            input.try_into()?,
            output.into(),
            Default::default(),
        ))
    }

    /// Moves an array to the gpu with the specified dimension.
    fn to_gpu(&self, data: &[f32], size: &[i64]) -> Result<Tensor, TchError> {
        Tensor::from(data).to_device(self.vs.device()).f_view_(size)
    }

    /// Performs one gradient descent iteration on a minibatch.
    fn train(&mut self, history: &[fxcm::CompleteTimestep], actions: u8) -> fxcm::Result<()> {
        let size = [history.len().try_into()?, actions.into()];
        let state = {
            let arr: ArrayVec<_, { cfg::BATCH * fxcm::Order::MAX }> = history
                .iter()
                .flat_map(|x| x.get_timestep().get_timestep().get_state(actions.into()))
                .collect();
            self.to_gpu(arr.as_slice(), &size)?
        };
        let action = {
            let arr: ArrayVec<_, { cfg::BATCH * fxcm::Order::MAX }> = history
                .iter()
                .flat_map(|x| x.get_timestep().get_action(actions.into()))
                .collect();
            self.to_gpu(arr.as_slice(), &size)?
        };
        let reward = {
            let arr: ArrayVec<_, { cfg::BATCH * fxcm::Order::MAX }> = history
                .iter()
                .flat_map(fxcm::CompleteTimestep::get_reward)
                .collect();
            self.to_gpu(arr.as_slice(), &size)?
        };
        let size = [
            history.len().try_into()?,
            fxcm::LAYER_DIM.try_into()?,
            cfg::FEATURES.try_into()?,
        ];
        let mut hidden = if cfg::STATEFUL {
            let arr: HiddenBatch = history
                .iter()
                .flat_map(|x| x.get_timestep().get_timestep().get_hidden())
                .collect();
            self.to_gpu(arr.as_slice(), &size)?
        } else {
            Tensor::f_zeros(&size, (Kind::Float, self.vs.device()))?
        };
        let n = history.len().try_into()?;
        let size = [
            n,
            cfg::WINDOW.try_into()?,
            2 * i64::try_from(fxcm::Symbol::N)?,
        ];
        let observation = {
            let arr: ObservationBatch = history
                .iter()
                .flat_map(|x| x.get_timestep().get_timestep().get_observation())
                .collect();
            self.to_gpu(arr.as_slice(), &size)?
        };
        let hidden = nn::GRUState(hidden.f_transpose_(0, 1)?);
        let observation = self.rnn.seq_init(&observation, &hidden);
        let observation = observation.0.f_flatten(1, -1)?;
        let state = Tensor::f_cat(&[&observation, &state], 1)?;
        let critic = self.critic.forward(&state);
        let actor = self.actor.forward(&state);
        let probs = actor.f_sigmoid()?;
        let action_log_probs = action.f_binary_cross_entropy_with_logits::<Tensor>(
            &actor,
            None,
            None,
            Reduction::Mean,
        )?;
        let dist_entropy = probs.f_binary_cross_entropy_with_logits::<Tensor>(
            &actor,
            None,
            None,
            Reduction::Mean,
        )?;
        let advantages = reward - critic;
        let value_loss = (&advantages * &advantages).mean(Kind::Float);
        let action_loss = (-advantages.detach() * action_log_probs).mean(Kind::Float);
        let loss = value_loss * 0.5 + action_loss - dist_entropy * 0.01;
        self.opt.backward_step_clip(&loss, 0.5);
        Ok(())
    }

    /// Learns from the historical data.
    pub fn update(
        &mut self,
        history: &mut [fxcm::CompleteTimestep],
        actions: u8,
    ) -> fxcm::Result<()> {
        for _ in 0..self.epochs {
            history.shuffle(&mut self.rng);
            for x in history.chunks(cfg::BATCH) {
                self.train(x, actions)?;
            }
        }
        Ok(())
    }

    /// Chooses whether to buy, hold or sell.
    pub fn act(
        &self,
        timestep: &mut fxcm::PartialTimestep,
        actions: u8,
    ) -> fxcm::Result<fxcm::Timestep> {
        let (action, hidden) = no_grad(|| self.choose(&*timestep, actions))?;
        Ok(timestep.act(action, hidden))
    }

    /// Chooses the actions based on the current actor network inference.
    fn choose(
        &self,
        timestep: &fxcm::PartialTimestep,
        actions: u8,
    ) -> fxcm::Result<(fxcm::State, fxcm::Hidden)> {
        const BATCH: usize = 1;
        let size = [
            BATCH.try_into()?,
            cfg::WINDOW.try_into()?,
            2 * i64::try_from(fxcm::Symbol::N)?,
        ];
        let observation = {
            let arr: ArrayVec<_, { BATCH * cfg::WINDOW * 2 * fxcm::Symbol::N }> =
                timestep.get_observation().collect();
            self.to_gpu(arr.as_slice(), &size)?
        };
        let size = [
            BATCH.try_into()?,
            fxcm::LAYER_DIM.try_into()?,
            cfg::FEATURES.try_into()?,
        ];
        let hidden = if cfg::STATEFUL {
            let arr: ArrayVec<_, { BATCH * fxcm::LAYER_DIM * cfg::FEATURES }> =
                timestep.get_hidden().collect();
            self.to_gpu(arr.as_slice(), &size)?
        } else {
            Tensor::f_zeros(&size, (Kind::Float, self.vs.device()))?
        };
        let hidden = nn::GRUState(hidden.f_transpose(0, 1)?);
        let (observation, nn::GRUState(hidden)) = self.rnn.seq_init(&observation, &hidden);
        let observation = observation.f_flatten(1, -1)?;
        let size = [BATCH.try_into()?, actions.into()];
        let state = {
            let arr: ArrayVec<_, { BATCH * fxcm::Order::MAX }> =
                timestep.get_state(actions.into()).collect();
            self.to_gpu(arr.as_slice(), &size)?
        };
        let state = Tensor::f_cat(&[&observation, &state], 1)?;
        let mut action = self.actor.forward(&state).f_unsqueeze_(0)?;
        let action = Vec::from(action.f_sigmoid_()?.f_bernoulli()?);
        let hidden = if cfg::STATEFUL {
            Vec::from(hidden).try_into()?
        } else {
            Default::default()
        };
        Ok((fxcm::State::from_slice(&action[..]), hidden))
    }
}
