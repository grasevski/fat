//! FXCM data model.
use arrayvec::ArrayVec;
use chrono::{DateTime, NaiveDateTime, ParseError, Utc};
use enum_map::Enum;
use num_derive::FromPrimitive;
use proptest_derive::Arbitrary;
use rust_decimal::prelude::{Decimal, One, ToPrimitive};
use serde::{Deserialize, Serialize};
use std::{cmp, fmt, io, num, result};
use strum_macros::{Display, EnumString};
use tch::TchError;

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

    /// Converts to a convenient format for machine learning.
    pub fn state(&self) -> impl Iterator<Item = self::Result<f32>> {
        [self.bid, self.ask]
            .iter()
            .map(|x| Ok(x.to_f64().ok_or(Error::F64(*x))? as f32))
            .collect::<ArrayVec<self::Result<f32>, 2>>()
            .into_iter()
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
    /// Invalid action arrayvec size.
    ArrayVec(ArrayVec<f32, { Order::MAX }>),

    /// Autotrader has fallen too far behind.
    CandleSet,

    /// Invalid candle arrayvec size.
    CandleVec,

    /// Parsing csv failed.
    Csv(csv::Error),

    /// Formatting a URL failed.
    Fmt(fmt::Error),

    /// Float conversion failed.
    F64(Decimal),

    /// Invalid initialization.
    Initialization,

    /// Error reading or writing to the exchange or data source.
    Io(io::Error),

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
}

impl From<ArrayVec<f32, { Order::MAX }>> for Error {
    fn from(error: ArrayVec<f32, { Order::MAX }>) -> Self {
        Self::ArrayVec(error)
    }
}

impl From<ArrayVec<Candle, { Symbol::N }>> for Error {
    fn from(_: ArrayVec<Candle, { Symbol::N }>) -> Self {
        Self::CandleVec
    }
}

impl From<csv::Error> for Error {
    fn from(error: csv::Error) -> Self {
        Self::Csv(error)
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

/// All available currencies.
#[derive(Clone, Copy, Debug, Deserialize, EnumString, PartialEq, Serialize)]
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
    Display,
    Deserialize,
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
    pub const N: usize = 21;

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
        Self {
            candle,
            balance: Default::default(),
        }
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
            let mut order = super::Order::new(usize::from(id) + 1, symbol, side, qty);
            let (b, q) = symbol.currencies();
            let currency = if base {b} else {q};
            assert_eq!(market.pnl(currency), Default::default());
            market.trade(&mut order);
            assert!(market.pnl(currency) <= Default::default());
            order.side = match order.side {
                super::Side::Bid => super::Side::Ask,
                super::Side::Ask => super::Side::Bid,
            };
            market.trade(&mut order);
            assert!(market.pnl(currency) <= Default::default());
        }
    }
}
