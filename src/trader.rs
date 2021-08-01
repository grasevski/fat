//! Trader interface and implementations.
use super::{cfg::ACTIONS, fxcm, model};
use arrayvec::ArrayVec;
use chrono::{DateTime, Utc};
use clap::Clap;
use enum_map::{Enum, EnumMap};
use rust_decimal::prelude::{Decimal, One, ToPrimitive};
use std::convert::TryInto;

/// A list of orders from the trader.
pub type OrderList = ArrayVec<fxcm::Order, { fxcm::Order::MAX }>;

/// An algorithmic trading strategy.
pub trait Trader {
    /// Runs when a candle is received.
    fn on_candle(&mut self, candle: fxcm::Candle) -> fxcm::Result<OrderList>;

    /// Runs when a trade is executed.
    fn on_order(&mut self, order: fxcm::Order) -> fxcm::Result<()>;
}

/// Dummy do nothing trader.
#[derive(Default)]
pub struct Dryrun {}

impl Trader for Dryrun {
    fn on_candle(&mut self, _: fxcm::Candle) -> fxcm::Result<OrderList> {
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

    /// Determines whether the current orders have completed.
    fn ready(&self) -> bool {
        self.qty == Default::default()
    }

    /// Updates the order status.
    fn update(&mut self, order: &fxcm::Order) {
        assert!(!self.ready());
        self.qty -= order.qty;
        assert!(self.ready());
        self.balance.update(order);
    }

    /// Calculates the delta in pnl.
    fn reward(&self, currency: fxcm::Currency, candle: &fxcm::Candle) -> Decimal {
        self.balance.pnl(currency, candle) - self.pnl
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
        assert!(self.ready());
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

/// Do nothing trader.
pub struct MrMagoo {
    /// Latest order sequence number.
    seq: usize,

    /// Settlement currency.
    currency: fxcm::Currency,

    /// Budget for each market.
    qty: Decimal,

    /// PPO agent.
    model: model::Model,

    /// State for each symbol being traded.
    markets: ArrayVec<Market, { fxcm::Order::MAX }>,

    /// State machine to batch up the candle updates.
    candle_aggregator: CandleAggregator,

    /// Market data.
    candles: Option<CandleSet>,

    /// Most recent observation data.
    partial: fxcm::PartialTimestep,

    /// Most recent action data.
    timestep: Option<fxcm::Timestep>,

    /// Historical activity.
    history: ArrayVec<fxcm::CompleteTimestep, { ACTIONS }>,
}

/// Runtime autotrader hyperparams.
#[derive(Clap)]
pub struct Cfg {
    /// Number of epochs per update.
    #[clap(short, long, default_value = "1")]
    iter: u8,

    /// Random number generator seed.
    #[clap(short, long, default_value = "0")]
    gen: i64,

    /// Dropout rate.
    #[clap(short, long, default_value = "0")]
    prob: f64,

    /// Learning rate.
    #[clap(short, long, default_value = "1e-3")]
    alpha: f64,

    /// Whether to exclude bias parameters from GRU layers.
    #[clap(short, long)]
    unbiased: bool,
}

impl MrMagoo {
    /// Initializes trader with settlement currency and bankroll.
    pub fn new(currency: fxcm::Currency, qty: Decimal, cfg: Cfg) -> fxcm::Result<Self> {
        let (seq, candles, partial, timestep, history) = Default::default();
        let markets = Market::new(currency);
        let n = markets.len().try_into()?;
        let model = model::Model::new(cfg.iter, cfg.gen, n, cfg.prob, cfg.alpha, !cfg.unbiased)?;
        Ok(Self {
            seq,
            currency,
            qty,
            model,
            markets,
            candle_aggregator: CandleAggregator::new()?,
            candles,
            partial,
            timestep,
            history,
        })
    }
}

impl Trader for MrMagoo {
    fn on_candle(&mut self, candle: fxcm::Candle) -> fxcm::Result<OrderList> {
        if let Some(curr) = self.candle_aggregator.next(candle)? {
            if self.candles.is_some() {
                return Err(fxcm::Error::CandleSet);
            }
            self.candles = Some(curr);
        }
        if let Some(candles) = self.candles.take() {
            if !self.markets.iter().all(Market::ready) {
                return Ok(Default::default());
            }
            let mut observation = EnumMap::default();
            for (o, x) in candles
                .values()
                .map(TryInto::try_into)
                .zip(observation.values_mut())
            {
                *x = o?;
            }
            self.partial.update(observation)?;
            let n = self.markets.len().try_into()?;
            if !self.partial.ready() {
                return Ok(Default::default());
            } else if let Some(timestep) = self.timestep.take() {
                let reward: fxcm::Result<fxcm::Reward> = self
                    .markets
                    .iter()
                    .map(|x| {
                        let ret = x.reward(self.currency, &candles[x.get_symbol()]) / self.qty;
                        Ok(ret.to_f64().ok_or(fxcm::Error::F64(ret))? as f32)
                    })
                    .collect();
                let timestep = fxcm::CompleteTimestep::new(reward?, timestep);
                self.history.try_push(timestep)?;
                if self.history.is_full() {
                    self.model.update(&mut self.history, n)?;
                    self.history.clear();
                }
            }
            let timestep = self.model.act(&mut self.partial, n)?;
            let (currency, qty) = (self.currency, self.qty);
            let ret = (self.seq..)
                .zip(timestep.action_bool(n.into()).zip(&mut self.markets))
                .filter_map(|(i, (a, m))| m.act(i, currency, &candles[m.get_symbol()], qty, a))
                .collect();
            self.timestep = Some(timestep);
            self.seq += self.markets.len();
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
