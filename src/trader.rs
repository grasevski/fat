//! Trader interface and implementations.
use super::fxcm;
use arrayvec::ArrayVec;
use enum_map::{Enum, EnumMap};
use rust_decimal::prelude::{Decimal, One, ToPrimitive};
use std::convert::{TryFrom, TryInto};
use tch::{
    nn,
    nn::{Module, OptimizerConfig},
    Device, Kind, Reduction, Tensor,
};

/// An algorithmic trading strategy.
pub trait Trader {
    /// Runs when a candle is received.
    fn on_candle(
        &mut self,
        candle: fxcm::Candle,
    ) -> fxcm::Result<ArrayVec<fxcm::Order, { fxcm::Order::MAX }>>;

    /// Runs when a trade is executed.
    fn on_order(&mut self, order: fxcm::Order) -> fxcm::Result<()>;
}

/// Minimal Gated Unit.
#[derive(Debug)]
struct Mgu<const H: i8> {
    /// Forget gate.
    f: nn::Linear,

    /// Hidden layer.
    h: nn::Linear,
}

impl<const H: i8> Mgu<H> {
    /// Initializes the MGU according to the input and hidden dimensions.
    fn new(path: &nn::Path, input: i8) -> Self {
        let n = i64::from(input + H);
        let f = nn::linear(path / "f", n, H.into(), Default::default());
        let h = nn::linear(path / "h", n, H.into(), Default::default());
        Self { f, h }
    }
}

impl<const H: i8> nn::Module for Mgu<H> {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let f = self.f.forward(xs).sigmoid_();
        let mut t = xs.tensor_split_indices(
            &[xs.size().last().expect("xs should not be empty") - i64::from(H)],
            (xs.dim() - 1)
                .try_into()
                .expect("invalid mgu input dimension"),
        );
        let h = self
            .h
            .forward(&Tensor::cat(
                &[&t[0], &(&f * &t[1])],
                (t[0].dim() - 1)
                    .try_into()
                    .expect("invalid mgu output dimension"),
            ))
            .tanh_();
        t[1] *= 1 - &f;
        t[1] += f * h;
        t[1].shallow_clone()
    }
}

/// All candles for a given time slice.
type CandleSet = EnumMap<fxcm::Symbol, fxcm::Candle>;

/// State machine to batch up candle updates.
struct CandleAggregator {
    /// Remaining candles to receive before outputting a batch.
    remaining: i8,

    /// Current batch of candles.
    candles: EnumMap<fxcm::Symbol, Option<fxcm::Candle>>,
}

impl CandleAggregator {
    /// Initializes the candle aggregator state.
    fn new() -> fxcm::Result<Self> {
        let candles = EnumMap::default();
        Ok(Self {
            remaining: candles.len().try_into()?,
            candles,
        })
    }

    /// Reads in the current candle and outputs a batch as necessary.
    fn next(&mut self, curr: fxcm::Candle) -> fxcm::Result<Option<CandleSet>> {
        let x = &mut self.candles[curr.symbol];
        if let Some(prev) = x {
            return Err(fxcm::Error::Candle {
                prev: prev.clone(),
                curr,
            });
        }
        *x = Some(curr);
        self.remaining -= 1;
        if self.remaining > 0 {
            return Ok(None);
        }
        let ret: fxcm::Result<ArrayVec<fxcm::Candle, { fxcm::Symbol::N }>> = self
            .candles
            .values()
            .map(|x| x.clone().ok_or(fxcm::Error::Initialization))
            .collect();
        let ret = CandleSet::from_array(ret?.into_inner()?);
        *self = Self::new()?;
        let ts = ret.values().next().ok_or(fxcm::Error::Initialization)?.ts;
        if ret.values().all(|x| x.ts == ts) {
            Ok(Some(ret))
        } else {
            Err(fxcm::Error::DateTime(ts))
        }
    }
}

/// State machine for the given symbol.
struct Market {
    /// The instrument being traded.
    symbol: fxcm::Symbol,

    /// Whether an investment has been made.
    position: bool,

    /// In flight volume.
    qty: Decimal,

    /// Current pnl, for reward calculations.
    pnl: Decimal,

    /// Tracks the current balance in each respective currency.
    balance: fxcm::Balance,
}

impl From<fxcm::Symbol> for Market {
    fn from(symbol: fxcm::Symbol) -> Self {
        Self {
            symbol,
            position: Default::default(),
            qty: Default::default(),
            pnl: Default::default(),
            balance: Default::default(),
        }
    }
}

impl Market {
    /// Initializes market state.
    fn new(currency: fxcm::Currency) -> ArrayVec<Self, { fxcm::Order::MAX }> {
        (0..fxcm::Symbol::N)
            .map(<fxcm::Symbol as Enum<()>>::from_usize)
            .filter(|x| x.has_currency(currency))
            .map(Into::into)
            .collect()
    }

    /// Returns the symbol for this market.
    fn get_symbol(&self) -> fxcm::Symbol {
        self.symbol
    }

    /// Returns the position in the market.
    fn get_position(&self) -> fxcm::Result<f32> {
        Ok(f32::from(i8::from(self.position)))
    }

    /// Determines whether the current orders have completed.
    fn ready(&self) -> bool {
        self.qty == Default::default()
    }

    /// Updates the order status.
    fn update(&mut self, order: &fxcm::Order) {
        self.qty -= order.qty;
        self.balance.update(order);
    }

    /// Calculates the delta in pnl.
    fn reward(&self, currency: fxcm::Currency, candle: &fxcm::Candle) -> fxcm::Result<f32> {
        let ret = self.balance.pnl(currency, candle) - self.pnl;
        Ok(ret.to_f64().ok_or(fxcm::Error::F64(ret))? as f32)
    }

    /// Assigns the next action.
    fn act(
        &mut self,
        seq: usize,
        currency: fxcm::Currency,
        candle: &fxcm::Candle,
        qty: Decimal,
        position: bool,
    ) -> Option<fxcm::Order> {
        self.pnl = self.balance.pnl(currency, candle);
        if self.position == position {
            return None;
        }
        let (side, quotient) = if self.symbol.currencies().0 == currency {
            let s = if position {
                fxcm::Side::Bid
            } else {
                fxcm::Side::Ask
            };
            (s, Decimal::one())
        } else if position {
            (fxcm::Side::Ask, candle.bid)
        } else {
            (fxcm::Side::Bid, candle.ask)
        };
        let order = fxcm::Order::new(seq, self.symbol, side, qty / quotient);
        self.position = !self.position;
        self.qty = order.qty;
        Some(order)
    }
}

/// Reinforcement learning agent.
struct Model<const N: usize, const H: i8> {
    /// PyTorch heap, may be on either GPU or CPU.
    vs: nn::VarStore,

    /// Gradient descent method.
    opt: nn::Optimizer<nn::AdamW>,

    /// RNN cell.
    mgu: Mgu<H>,

    /// Which position to hold for each symbol.
    actor: nn::Linear,

    /// Value of the current state.
    critic: nn::Linear,

    /// Hidden state.
    hidden: Tensor,

    /// Actor output.
    actor_out: Tensor,

    /// Actor output probabilities.
    probs: Tensor,

    /// Randomly chosen action.
    action: Tensor,
}

impl<const N: usize, const H: i8> Model<N, H> {
    /// Initialize model with number of symbols and number of active markets.
    fn new(o: i8) -> fxcm::Result<Self> {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let opt = nn::AdamW::default().build(&vs, 1e-3)?;
        let mgu = Mgu::new(&(&vs.root() / "mgu"), 2 * i8::try_from(N)? + o);
        let actor = nn::linear(&vs.root() / "actor", H.into(), o.into(), Default::default());
        let critic = nn::linear(
            &vs.root() / "critic",
            H.into(),
            o.into(),
            Default::default(),
        );
        let hidden = vs.root().f_zeros("hidden", &[H.into()])?;
        let actor_out = vs.root().f_zeros("actor_out", &[o.into()])?;
        let probs = vs.root().f_zeros("probs", &[o.into()])?;
        let action = vs.root().f_zeros("action", &[o.into()])?;
        Ok(Self {
            vs,
            opt,
            mgu,
            actor,
            critic,
            hidden,
            actor_out,
            probs,
            action,
        })
    }

    /// Learn from current state and choose next action.
    fn act(&mut self, reward: &[f32], state: &[f32]) -> fxcm::Result<Vec<bool>> {
        let action_log_probs = self.action.f_binary_cross_entropy_with_logits::<Tensor>(
            &self.actor_out,
            None,
            None,
            Reduction::Mean,
        )?;
        let dist_entropy = self.probs.f_binary_cross_entropy_with_logits::<Tensor>(
            &self.actor_out,
            None,
            None,
            Reduction::Mean,
        )?;
        let reward = Tensor::from(reward).to_device(self.vs.device());
        let state = Tensor::from(state).to_device(self.vs.device());
        let advantages = reward - self.critic.forward(&self.hidden);
        let value_loss = (&advantages * &advantages).f_mean(Kind::Float)?;
        let action_loss = (-advantages.detach() * action_log_probs).f_mean(Kind::Float)?;
        let loss = value_loss * 0.5 + action_loss - dist_entropy * 0.01;
        self.opt.backward_step_clip(&loss, 0.5);
        self.hidden = self.mgu.forward(&Tensor::cat(
            &[&state, &self.hidden],
            (state.dim() - 1).try_into()?,
        ));
        self.actor_out = self.actor.forward(&self.hidden);
        self.probs = self.actor_out.f_sigmoid()?;
        self.action = self.probs.f_bernoulli()?;
        Ok(Vec::from(&self.action))
    }
}

/// Do nothing trader.
pub struct MrMagoo {
    /// Latest order sequence number.
    seq: usize,

    /// Settlement currency.
    currency: fxcm::Currency,

    /// State for each symbol being traded.
    markets: ArrayVec<Market, { fxcm::Order::MAX }>,

    /// State machine to batch up the candle updates.
    candle_aggregator: CandleAggregator,

    /// Market data.
    candles: Option<CandleSet>,

    /// Budget for each market.
    qty: Decimal,

    /// Training iterations remaining.
    train: i16,

    /// PPO agent.
    model: Model<{ fxcm::Symbol::N }, 16>,
}

impl MrMagoo {
    /// Initializes trader with settlement currency and bankroll.
    pub fn new(currency: fxcm::Currency, qty: Decimal, train: i16) -> fxcm::Result<Self> {
        let (seq, candles) = Default::default();
        let markets = Market::new(currency);
        let candle_aggregator = CandleAggregator::new()?;
        let model = Model::new(markets.len().try_into()?)?;
        Ok(Self {
            seq,
            currency,
            markets,
            candle_aggregator,
            candles,
            qty,
            train,
            model,
        })
    }
}

impl Trader for MrMagoo {
    fn on_candle(
        &mut self,
        candle: fxcm::Candle,
    ) -> fxcm::Result<ArrayVec<fxcm::Order, { fxcm::Order::MAX }>> {
        if self.train > 0 {
            self.train -= 1;
            if self.train == 0 {
                self.markets = Market::new(self.currency);
            }
        }
        if let Some(curr) = self.candle_aggregator.next(candle)? {
            if self.candles.is_some() {
                return Err(fxcm::Error::CandleSet);
            }
            self.candles = Some(curr);
        }
        if let Some(candles) = &self.candles {
            if !self.markets.iter().all(Market::ready) {
                return Ok(Default::default());
            }
            let reward: fxcm::Result<ArrayVec<f32, { fxcm::Order::MAX }>> = self
                .markets
                .iter()
                .map(|x| x.reward(self.currency, &candles[x.get_symbol()]))
                .collect();
            let state: fxcm::Result<ArrayVec<f32, { fxcm::Order::MAX }>> = candles
                .values()
                .flat_map(fxcm::Candle::state)
                .chain(self.markets.iter().map(Market::get_position))
                .collect();
            let action = self.model.act(reward?.as_slice(), state?.as_slice())?;
            let (currency, qty) = (self.currency, self.qty);
            let ret = (self.seq..)
                .zip(action.into_iter().zip(&mut self.markets))
                .filter_map(|(i, (a, m))| m.act(i, currency, &candles[m.get_symbol()], qty, a))
                .collect();
            self.seq += self.markets.len();
            Ok(ret)
        } else {
            Ok(Default::default())
        }
    }

    fn on_order(&mut self, order: fxcm::Order) -> fxcm::Result<()> {
        let n = self.markets.len();
        self.markets[order.id - self.seq - n].update(&order);
        Ok(())
    }
}
