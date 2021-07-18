//! FXCM data model.
use super::cfg;
use arrayvec::{ArrayVec, CapacityError};
use bitvec::BitArr;
use chrono::{DateTime, NaiveDateTime, ParseError, Utc};
use enum_map::{Enum, EnumMap};
use num_derive::FromPrimitive;
use proptest_derive::Arbitrary;
use rust_decimal::prelude::{Decimal, One};
use serde::{Deserialize, Serialize};
use static_assertions::const_assert;
use std::{cmp, convert::TryFrom, fmt, io, mem, num, result};
use strum_macros::{Display, EnumString};
use tch::TchError;

const_assert!(cfg::LAYERS != 0);
const_assert!(cfg::FEATURES != 0);
const_assert!(cfg::SEQ_LEN != 0);
const_assert!(cfg::STEPS != 0);
const_assert!(!cfg::STATEFUL || !cfg::BIDIRECTIONAL);

/// Profit and loss for each market.
pub type Reward = ArrayVec<f32, { Order::MAX }>;

/// Dimension for the hidden variables.
pub const LAYER_DIM: usize = cfg::LAYERS * if cfg::BIDIRECTIONAL { 2 } else { 1 };

/// Hidden state size.
const HIDDEN: usize = if cfg::STATEFUL {
    LAYER_DIM * cfg::FEATURES
} else {
    0
};

/// Hidden state vector.
#[derive(Clone)]
pub struct Hidden([f32; HIDDEN]);

impl Default for Hidden {
    fn default() -> Self {
        Self([Default::default(); HIDDEN])
    }
}

impl TryFrom<Vec<f32>> for Hidden {
    type Error = self::Error;

    fn try_from(hidden: Vec<f32>) -> self::Result<Self> {
        Ok(Self(TryFrom::try_from(hidden)?))
    }
}

impl From<&Hidden> for ArrayVec<f32, HIDDEN> {
    fn from(hidden: &Hidden) -> Self {
        From::from(hidden.0)
    }
}

/// State vector representation.
type StateVec = BitArr!(for Order::MAX, in u8);

/// Trader position representation.
#[derive(Clone, Copy, Default)]
pub struct State(StateVec);

impl State {
    /// Converts the slice into a compact state.
    pub fn from_slice(xs: &[bool]) -> Self {
        let mut data = StateVec::zeroed();
        for (mut d, x) in data.iter_mut().zip(xs) {
            *d = *x;
        }
        Self(data)
    }

    /// Converts the state into a tensor format.
    fn into_iter_f32(self, n: usize) -> impl Iterator<Item = f32> {
        self.into_iter(n).map(u16::from).map(f32::from)
    }

    /// Converts the state into a bool iterator.
    fn into_iter(self, n: usize) -> impl Iterator<Item = bool> {
        self.0.into_iter().take(n)
    }
}

/// All bid and ask prices for a given time slice.
pub type ObservationSet = EnumMap<Symbol, Observation>;

/// Market data for one symbol for one timestep.
#[derive(Default)]
pub struct Observation {
    /// Highest bid.
    bid: f32,

    /// Lowest ask.
    ask: f32,
}

impl TryFrom<&Candle> for Observation {
    type Error = self::Error;

    fn try_from(candle: &Candle) -> self::Result<Self> {
        let (bid, ask) = (f32::try_from(candle.bid)?, f32::try_from(candle.ask)?);
        Ok(Self { bid, ask })
    }
}

impl Observation {
    /// Returns an iterator for the data in this observation.
    fn iter(&self) -> impl Iterator<Item = f32> {
        ArrayVec::from([self.bid, self.ask]).into_iter()
    }
}

/// The input part of the state.
#[derive(Default)]
pub struct PartialTimestep {
    /// Trader state.
    state: State,

    /// Hidden state leading into this step.
    hidden: Hidden,

    /// The preceding sequence of observations.
    observation: ArrayVec<ObservationSet, { cfg::SEQ_LEN }>,
}

impl PartialTimestep {
    /// Adds another observation to the sequence.
    pub fn update(&mut self, observation: ObservationSet) -> self::Result<()> {
        self.observation.try_push(observation)?;
        Ok(())
    }

    /// Determines whether we are ready for action.
    pub fn ready(&self) -> bool {
        self.observation.is_full()
    }

    /// Outputs timestep info and updates state machine.
    pub fn act(&mut self, action: State, mut hidden: Hidden) -> Timestep {
        if !cfg::STATEFUL {
            hidden = self.hidden.clone();
        }
        let mut timestep = Self {
            state: action,
            hidden,
            observation: Default::default(),
        };
        mem::swap(self, &mut timestep);
        Timestep { action, timestep }
    }

    /// Returns an iterator over the state data.
    pub fn get_state(&self, actions: usize) -> impl Iterator<Item = f32> {
        self.state.into_iter_f32(actions)
    }

    /// Returns an iterator over the hidden data.
    pub fn get_hidden(&self) -> impl Iterator<Item = f32> {
        ArrayVec::from(&self.hidden).into_iter()
    }

    /// Returns an iterator over the observation data.
    pub fn get_observation(&self) -> impl Iterator<Item = f32> {
        let ret: ArrayVec<_, { cfg::SEQ_LEN * 2 * Symbol::N }> = self
            .observation
            .iter()
            .flat_map(|x| x.values().flat_map(Observation::iter))
            .collect();
        ret.into_iter()
    }
}

/// The action plus the input part of the state.
pub struct Timestep {
    /// Action taken by the trader.
    action: State,

    /// Other fields.
    timestep: PartialTimestep,
}

impl Timestep {
    /// Returns an iterator of booleans over the actions taken.
    pub fn action_bool(&self, actions: usize) -> impl Iterator<Item = bool> {
        self.action.into_iter(actions)
    }

    /// Returns an iterator over the actions taken.
    pub fn get_action(&self, actions: usize) -> impl Iterator<Item = f32> {
        self.action.into_iter_f32(actions)
    }

    /// Accesses the other timestep information.
    pub fn get_timestep(&self) -> &PartialTimestep {
        &self.timestep
    }
}

/// A completed historical record.
pub struct CompleteTimestep {
    /// Profit and loss from this step.
    reward: Reward,

    /// State data.
    timestep: Timestep,
}

impl CompleteTimestep {
    /// Constructs the completed historical record.
    pub fn new(reward: Reward, timestep: Timestep) -> Self {
        Self { reward, timestep }
    }

    /// Returns an iterator over the reward data.
    pub fn get_reward(&self) -> impl Iterator<Item = f32> {
        self.reward.clone().into_iter()
    }

    /// Accesses the other timestep information.
    pub fn get_timestep(&self) -> &Timestep {
        &self.timestep
    }
}

/// Candle iterators may fail and return an error instead.
pub type FallibleCandle = self::Result<Candle>;

/// Historical candle data is a slightly different format.
#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct Historical {
    /// Timestamp in US format.
    date_time: String,

    /// Opening bid price for time period.
    bid_open: Decimal,

    /// Opening ask price for time period.
    ask_open: Decimal,
}

/// Candle data from the exchange.
#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct Candle {
    /// When the data occurred.
    pub ts: DateTime<Utc>,

    /// The instrument for the given record.
    pub symbol: Symbol,

    /// Latest bid price.
    pub bid: Decimal,

    /// Latest ask price.
    pub ask: Decimal,
}

impl Candle {
    /// Converts a historical data candle into a regular candle.
    pub fn new(symbol: Symbol, historical: Historical) -> self::Result<Self> {
        assert_ne!(historical.bid_open, Default::default());
        assert_ne!(historical.ask_open, Default::default());
        assert!(historical.bid_open > Default::default());
        assert!(historical.ask_open > Default::default());
        let ts = NaiveDateTime::parse_from_str(&historical.date_time, "%m/%d/%Y %H:%M:%S%.f")?;
        let ts = DateTime::from_utc(ts, Utc);
        Ok(Self {
            symbol,
            ts,
            bid: historical.bid_open,
            ask: historical.ask_open,
        })
    }
}

/// Data type for inserting and executing orders.
#[derive(Debug, Serialize)]
pub struct Order {
    /// Unique identifier.
    pub id: usize,

    /// Insertion and or execution time.
    pub ts: DateTime<Utc>,

    /// Instrument or market.
    pub symbol: Symbol,

    /// Which side of the orderbook this order is entering.
    pub side: Side,

    /// Execution price.
    pub price: Decimal,

    /// Insertion or execution quantity.
    pub qty: Decimal,
}

/// Returns a placeholder epoch timestamp before proper initialization.
fn dummy_timestamp() -> DateTime<Utc> {
    DateTime::from_utc(NaiveDateTime::from_timestamp(0, 0), Utc)
}

impl Order {
    /// Maximum number of orders that can be inserted at a time.
    pub const MAX: usize = 8;

    /// Creates a new order to be inserted by the trader.
    pub fn new(id: usize, symbol: Symbol, side: Side, qty: Decimal) -> Self {
        assert_ne!(qty, Default::default());
        assert!(qty > Default::default());
        Self {
            id,
            ts: dummy_timestamp(),
            symbol,
            side,
            price: Default::default(),
            qty,
        }
    }
}

/// Whether it is a buy order or a sell order.
#[derive(Arbitrary, Debug, Serialize)]
pub enum Side {
    /// A buy order.
    Bid,

    /// A sell order.
    Ask,
}

/// All possible events that can be received from the exchange.
pub enum Event {
    /// A periodic candle datum.
    Candle(Candle),

    /// An order created by the trader.
    Order(Order),
}

/// FXCM specific result type.
pub type Result<T> = result::Result<T, self::Error>;

/// All possible errors returned by this library.
#[derive(Debug)]
pub enum Error {
    /// Autotrader has fallen too far behind.
    CandleSet,

    /// Invalid candle arrayvec size.
    CandleVec,

    /// History buffer overflow.
    CompleteTimestep,

    /// Parsing csv failed.
    Csv(csv::Error),

    /// Decimal conversion failed.
    Decimal(rust_decimal::Error),

    /// Formatting a URL failed.
    Fmt(fmt::Error),

    /// Float conversion failed.
    F64(Decimal),

    /// Invalid initialization.
    Initialization,

    /// Error reading or writing to the exchange or data source.
    Io(io::Error),

    /// Observation buffer overflow.
    ObservationSet(CapacityError<ObservationSet>),

    /// Too many orders sent to the exchange.
    Order(Order),

    /// Failed to parse timestamp.
    ParseError(ParseError),

    /// HTTP request failed.
    Reqwest(reqwest::Error),

    /// PyTorch exception.
    Tch(TchError),

    /// Integer type conversion failed.
    TryFromInt(num::TryFromIntError),

    /// State type conversion failed.
    Vec(Vec<f32>),
}

impl From<ArrayVec<Candle, { Symbol::N }>> for Error {
    fn from(_: ArrayVec<Candle, { Symbol::N }>) -> Self {
        Self::CandleVec
    }
}

impl From<CapacityError<CompleteTimestep>> for Error {
    fn from(_: CapacityError<CompleteTimestep>) -> Self {
        Self::CompleteTimestep
    }
}

impl From<CapacityError<ObservationSet>> for Error {
    fn from(error: CapacityError<ObservationSet>) -> Self {
        Self::ObservationSet(error)
    }
}

impl From<csv::Error> for Error {
    fn from(error: csv::Error) -> Self {
        Self::Csv(error)
    }
}

impl From<rust_decimal::Error> for Error {
    fn from(error: rust_decimal::Error) -> Self {
        Self::Decimal(error)
    }
}

impl From<fmt::Error> for Error {
    fn from(error: fmt::Error) -> Self {
        Self::Fmt(error)
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<Order> for Error {
    fn from(order: Order) -> Self {
        Self::Order(order)
    }
}

impl From<ParseError> for Error {
    fn from(error: ParseError) -> Self {
        Self::ParseError(error)
    }
}

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        Self::Reqwest(error)
    }
}

impl From<TchError> for Error {
    fn from(error: TchError) -> Self {
        Self::Tch(error)
    }
}

impl From<num::TryFromIntError> for Error {
    fn from(error: num::TryFromIntError) -> Self {
        Self::TryFromInt(error)
    }
}

impl From<Vec<f32>> for Error {
    fn from(error: Vec<f32>) -> Self {
        Self::Vec(error)
    }
}

/// All available currencies.
#[derive(Clone, Copy, Debug, Deserialize, Display, Enum, EnumString, PartialEq, Serialize)]
#[serde(rename_all = "UPPERCASE")]
#[strum(serialize_all = "UPPERCASE")]
pub enum Currency {
    /// Australian Dollar.
    Aud,

    /// Canadian Dollar.
    Cad,

    /// Swiss Franc.
    Chf,

    /// Euro.
    Eur,

    /// Great British Pound.
    Gbp,

    /// Japanese Yen.
    Jpy,

    /// New Zealand Dollar.
    Nzd,

    /// United States Dollar.
    Usd,
}

/// All available symbols.
#[derive(
    Arbitrary,
    Clone,
    Copy,
    Debug,
    Deserialize,
    Display,
    Enum,
    Eq,
    FromPrimitive,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
)]
#[serde(rename_all = "UPPERCASE")]
#[strum(serialize_all = "UPPERCASE")]
pub enum Symbol {
    AudCad,
    AudChf,
    AudJpy,
    AudNzd,
    CadChf,
    EurAud,
    EurChf,
    EurGbp,
    EurJpy,
    EurUsd,
    GbpChf,
    GbpJpy,
    GbpNzd,
    GbpUsd,
    NzdCad,
    NzdChf,
    NzdJpy,
    NzdUsd,
    UsdCad,
    UsdChf,
    UsdJpy,
}

impl Symbol {
    /// Total number of symbols.
    pub const N: usize = mem::size_of::<EnumMap<Symbol, u8>>();

    /// Returns whether the given currency is related to this symbol.
    pub fn has_currency(self, currency: Currency) -> bool {
        let (base, quote) = self.currencies();
        currency == base || currency == quote
    }

    /// Returns the base and quote currency for the given symbol.
    pub fn currencies(self) -> (Currency, Currency) {
        let ret = match self {
            Self::AudCad => (Currency::Aud, Currency::Cad),
            Self::AudChf => (Currency::Aud, Currency::Chf),
            Self::AudJpy => (Currency::Aud, Currency::Jpy),
            Self::AudNzd => (Currency::Aud, Currency::Nzd),
            Self::CadChf => (Currency::Cad, Currency::Chf),
            Self::EurAud => (Currency::Eur, Currency::Aud),
            Self::EurChf => (Currency::Eur, Currency::Chf),
            Self::EurGbp => (Currency::Eur, Currency::Gbp),
            Self::EurJpy => (Currency::Eur, Currency::Jpy),
            Self::EurUsd => (Currency::Eur, Currency::Usd),
            Self::GbpChf => (Currency::Gbp, Currency::Chf),
            Self::GbpJpy => (Currency::Gbp, Currency::Jpy),
            Self::GbpNzd => (Currency::Gbp, Currency::Nzd),
            Self::GbpUsd => (Currency::Gbp, Currency::Usd),
            Self::NzdCad => (Currency::Nzd, Currency::Cad),
            Self::NzdChf => (Currency::Nzd, Currency::Chf),
            Self::NzdJpy => (Currency::Nzd, Currency::Jpy),
            Self::NzdUsd => (Currency::Nzd, Currency::Usd),
            Self::UsdCad => (Currency::Usd, Currency::Cad),
            Self::UsdChf => (Currency::Usd, Currency::Chf),
            Self::UsdJpy => (Currency::Usd, Currency::Jpy),
        };
        assert_ne!(ret.0, ret.1);
        ret
    }
}

/// Per symbol balance.
#[derive(Default)]
pub struct Balance {
    /// Trader balance for the base currency of the given symbol.
    base: Decimal,

    /// Trader balance for the quote currency of the given symbol.
    quote: Decimal,
}

impl Balance {
    /// Incorporates executed order into balance.
    pub fn update(&mut self, order: &Order) {
        let qty = order.qty
            * match order.side {
                Side::Bid => Decimal::one(),
                Side::Ask => -Decimal::one(),
            };
        self.base -= qty;
        self.quote += qty / order.price;
    }

    /// Calculates the pnl with respect to the given currency and candle.
    pub fn pnl(&self, currency: Currency, candle: &Candle) -> Decimal {
        let (base, quote) = candle.symbol.currencies();
        if currency == base {
            let p = if self.quote > Default::default() {
                cmp::min
            } else {
                cmp::max
            };
            self.base + self.quote * p(candle.bid, candle.ask)
        } else if currency == quote {
            let p = if self.base > Default::default() {
                cmp::max
            } else {
                cmp::min
            };
            self.quote + self.base / p(candle.bid, candle.ask)
        } else {
            assert_eq!(self.base, Default::default());
            assert_eq!(self.quote, Default::default());
            Default::default()
        }
    }
}

/// Tracks the current position and pnl for a given symbol.
pub struct Market {
    /// Current candle data.
    candle: Candle,

    /// Movement of funds on this instrument.
    balance: Balance,
}

impl From<Candle> for Market {
    fn from(candle: Candle) -> Self {
        let balance = Default::default();
        Self { candle, balance }
    }
}

impl Market {
    /// Updates to latest candle data.
    pub fn update(&mut self, candle: Candle) {
        self.candle = candle;
    }

    /// Updates the current position based on the given trade.
    pub fn trade(&mut self, order: &mut Order) {
        order.price = match order.side {
            Side::Bid => cmp::max,
            Side::Ask => cmp::min,
        }(self.candle.bid, self.candle.ask);
        self.balance.update(order);
    }

    /// Calculates the pnl with respect to the given currency.
    pub fn pnl(&self, currency: Currency) -> Decimal {
        self.balance.pnl(currency, &self.candle)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::proptest;
    use rust_decimal::prelude::{Decimal, One};

    proptest! {
        #[test]
        fn crossing_the_spread_costs_money(symbol: super::Symbol, bid: u8, ask: u8, base: bool, id: u8, side: super::Side, qty: u8) {
            let q = Decimal::from(16);
            let (mut bid, mut ask, mut qty) = (Decimal::from(bid), Decimal::from(ask), Decimal::from(qty));
            bid += Decimal::one();
            ask += Decimal::one();
            qty += Decimal::one();
            bid /= q;
            ask /= q;
            qty /= q;
            let candle = super::Candle{ts: super::dummy_timestamp(), symbol, bid, ask};
            let mut market = super::Market::from(candle);
            let mut order = super::Order::new(id.into(), symbol, side, qty);
            let (b, q) = symbol.currencies();
            let currency = if base {b} else {q};
            assert_eq!(market.pnl(currency), Default::default());
            market.trade(&mut order);
            if bid == ask {
                assert_eq!(market.pnl(currency), Default::default());
            } else {
                assert!(market.pnl(currency) < Default::default());
            }
            order.side = match order.side {
                super::Side::Bid => super::Side::Ask,
                super::Side::Ask => super::Side::Bid,
            };
            market.trade(&mut order);
            if bid == ask {
                assert_eq!(market.pnl(currency), Default::default());
            } else {
                assert!(market.pnl(currency) < Default::default());
            }
        }
    }
}
