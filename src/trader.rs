//! Trader interface and implementations.
use super::fxcm;
use arrayvec::ArrayVec;
use chrono::{DateTime, Utc};
use enum_map::{Enum, EnumMap};
use rust_decimal::prelude::{Decimal, One, ToPrimitive};
use std::convert::{TryFrom, TryInto};
use tch::{
    manual_seed, nn,
    nn::{Module, RNN},
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

/// Dummy do nothing trader.
#[derive(Default)]
pub struct Dryrun {}

impl Trader for Dryrun {
    fn on_candle(
        &mut self,
        _: fxcm::Candle,
    ) -> fxcm::Result<ArrayVec<fxcm::Order, { fxcm::Order::MAX }>> {
        Ok(Default::default())
    }

    fn on_order(&mut self, _: fxcm::Order) -> fxcm::Result<()> {
        Ok(())
    }
}

/// All candles for a given time slice.
type CandleSet = EnumMap<fxcm::Symbol, fxcm::Candle>;

/// State machine to batch up candle updates.
struct CandleAggregator {
    /// Remaining candles for current batch.
    remaining: u8,

    /// Current timestamp.
    watermark: Option<DateTime<Utc>>,

    /// Current batch of candles.
    candles: EnumMap<fxcm::Symbol, Option<fxcm::Candle>>,
}

impl CandleAggregator {
    fn new() -> fxcm::Result<Self> {
        let remaining = fxcm::Symbol::N.try_into()?;
        let (watermark, candles) = Default::default();
        Ok(Self {
            remaining,
            watermark,
            candles,
        })
    }

    /// Reads in the current candle and outputs a batch as necessary.
    fn next(&mut self, candle: fxcm::Candle) -> fxcm::Result<Option<CandleSet>> {
        let symbol = candle.symbol;
        if let Some(ref mut watermark) = self.watermark {
            self.remaining -= 1;
            if candle.ts < *watermark || (candle.ts == *watermark && self.remaining > 0) {
                self.candles[symbol] = Some(candle);
                return Ok(None);
            }
            let ret: fxcm::Result<ArrayVec<fxcm::Candle, { fxcm::Symbol::N }>> = self
                .candles
                .values()
                .map(|x| x.clone().ok_or(fxcm::Error::Initialization))
                .collect();
            self.remaining = fxcm::Symbol::N.try_into()?;
            self.remaining -= u8::from(candle.ts == *watermark);
            *watermark = candle.ts;
            self.candles[symbol] = Some(candle);
            return Ok(Some(CandleSet::from_array(ret?.into_inner()?)));
        } else if self.candles.values().all(Option::is_some) {
            let watermark = self
                .candles
                .values()
                .filter_map(|x| x.as_ref().map(|c| c.ts))
                .max()
                .ok_or(fxcm::Error::Initialization)?;
            self.watermark = Some(watermark);
        };
        self.candles[symbol] = Some(candle);
        Ok(None)
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
struct Model<O: nn::OptimizerConfig, const N: usize, const H: u8> {
    /// PyTorch heap, may be on either GPU or CPU.
    vs: nn::VarStore,

    /// Gradient descent method.
    opt: nn::Optimizer<O>,

    /// Stateful time series decoder.
    rnn: nn::GRU,

    /// Which position to hold for each symbol.
    actor: nn::Sequential,

    /// Value of the current state.
    critic: nn::Sequential,

    /// Hidden state.
    hidden: nn::GRUState,

    /// Actor output.
    actor_out: Tensor,

    /// Actor output probabilities.
    probs: Tensor,

    /// Randomly chosen action.
    action: Tensor,
}

impl<O: nn::OptimizerConfig, const N: usize, const H: u8> Model<O, N, H> {
    /// Initialize model with number of symbols and number of active markets.
    fn new(cfg: O, o: i8) -> fxcm::Result<Self> {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let opt = cfg.build(&vs, 1e-3)?;
        let cfg = nn::RNNConfig {
            num_layers: 1,
            dropout: 0.0,
            ..Default::default()
        };
        let rnn = nn::gru(
            &(&vs.root() / "rnn"),
            2 * i64::try_from(N)? + i64::from(o),
            H.into(),
            cfg,
        );
        let actor = Self::predictor(&vs.root() / "actor", cfg.num_layers, o);
        let critic = Self::predictor(&vs.root() / "critic", cfg.num_layers, o);
        let hidden = rnn.zero_state(1);
        let (actor_out, probs, action) = Default::default();
        Ok(Self {
            vs,
            opt,
            rnn,
            actor,
            critic,
            hidden,
            actor_out,
            probs,
            action,
        })
    }

    /// Prediction head multi layer perceptron.
    fn predictor(path: nn::Path, num_layers: i64, o: i8) -> nn::Sequential {
        let fc = nn::linear(
            path,
            num_layers * i64::from(H),
            o.into(),
            Default::default(),
        );
        nn::seq().add_fn(|xs| xs.flatten(0, -1)).add(fc)
    }

    /// Learn from current state and choose next action.
    fn act(&mut self, reward: &[f32], state: &[f32]) -> fxcm::Result<Vec<bool>> {
        if self.probs.defined() {
            let reward = Tensor::from(reward).to_device(self.vs.device());
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
            let advantages = reward - self.critic.forward(&self.hidden.0);
            let value_loss = (&advantages * &advantages).f_mean(Kind::Float)?;
            let action_loss = (-advantages.detach() * action_log_probs).f_mean(Kind::Float)?;
            print!("\r{:?} {:?} {:?}", &value_loss, &action_loss, &dist_entropy);
            let loss = value_loss * 0.5 + action_loss - dist_entropy * 0.01;
            self.opt.backward_step_clip(&loss, 0.5);
        }
        let state = Tensor::from(state).unsqueeze(0).to_device(self.vs.device());
        self.hidden.0 = self.hidden.0.f_detach_()?;
        self.hidden = self.rnn.step(&state, &self.hidden);
        self.actor_out = self.actor.forward(&self.hidden.0);
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
    train: i32,

    /// PPO agent.
    model: Model<nn::RmsProp, { fxcm::Symbol::N }, 1>,
}

impl MrMagoo {
    /// Initializes trader with settlement currency and bankroll.
    pub fn new(
        currency: fxcm::Currency,
        qty: Decimal,
        train: i32,
        seed: i64,
    ) -> fxcm::Result<Self> {
        let (seq, candles) = Default::default();
        let markets = Market::new(currency);
        manual_seed(seed);
        let model = Model::new(Default::default(), markets.len().try_into()?)?;
        Ok(Self {
            seq,
            currency,
            markets,
            candle_aggregator: CandleAggregator::new()?,
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
        if let Some(candles) = self.candles.clone() {
            if !self.markets.iter().all(Market::ready) {
                return Ok(Default::default());
            }
            let reward: fxcm::Result<ArrayVec<f32, { fxcm::Order::MAX }>> = self
                .markets
                .iter()
                .map(|x| x.reward(self.currency, &candles[x.get_symbol()]))
                .collect();
            let state: fxcm::Result<ArrayVec<f32, { 2 * fxcm::Symbol::N + fxcm::Order::MAX }>> =
                candles
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
            self.candles = None;
            Ok(ret)
        } else {
            Ok(Default::default())
        }
    }

    fn on_order(&mut self, order: fxcm::Order) -> fxcm::Result<()> {
        let n = self.markets.len();
        self.markets[order.id + n - self.seq].update(&order);
        Ok(())
    }
}
