//! FXCM data model.
use chrono::{DateTime, NaiveDateTime, ParseError, Utc};
use enum_map::Enum;
use num_derive::FromPrimitive;
use proptest_derive::Arbitrary;
use rust_decimal::prelude::{Decimal, One};
use serde::{Deserialize, Serialize};
use std::{fmt, io, num, result};
use strum_macros::{Display, EnumString};

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
#[derive(Clone, Copy, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
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
        assert!(historical.ask_open > historical.bid_open);
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

fn dummy_timestamp() -> DateTime<Utc> {
    DateTime::from_utc(NaiveDateTime::from_timestamp(0, 0), Utc)
}

impl Order {
    /// Creates a new order to be inserted by the trader.
    pub fn new(id: usize, symbol: Symbol, side: Side, qty: Decimal) -> Option<Result<Self>> {
        assert_ne!(id, Default::default());
        assert_ne!(qty, Default::default());
        assert!(qty > Default::default());
        let price = Default::default();
        Some(Ok(Self {
            id,
            ts: dummy_timestamp(),
            symbol,
            side,
            price,
            qty,
        }))
    }
}

/// Whether it is a buy order or a sell order.
#[derive(Arbitrary, Debug, Serialize)]
pub enum Side {
    Bid,
    Ask,
}

/// All possible events that can be received from the exchange.
pub enum Event {
    Candle(Candle),
    Order(Order),
}

pub type Result<T> = result::Result<T, self::Error>;

/// All possible errors returned by this library.
#[derive(Debug)]
pub enum Error {
    Csv(csv::Error),
    Fmt(fmt::Error),
    Initialization,
    Io(io::Error),
    Order(Order),
    ParseError(ParseError),
    Reqwest(reqwest::Error),
    TryFromInt(num::TryFromIntError),
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
    Aud,
    Cad,
    Chf,
    Eur,
    Gbp,
    Jpy,
    Nzd,
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
    fn currencies(self) -> (Currency, Currency) {
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

/// Tracks the current position and pnl for a given symbol.
pub struct Market {
    candle: Candle,
    base_balance: Decimal,
    quote_balance: Decimal,
}

impl From<Candle> for Market {
    fn from(candle: Candle) -> Self {
        Self {
            candle,
            base_balance: Default::default(),
            quote_balance: Default::default(),
        }
    }
}

impl Market {
    /// Updates to latest candle data.
    pub fn update(&mut self, candle: Candle) {
        self.candle = candle;
    }

    /// Calculates the pnl with respect to the given currency.
    pub fn pnl(&self, currency: Currency) -> Decimal {
        let (base, quote) = self.candle.symbol.currencies();
        if currency == base {
            let p = if self.quote_balance > Default::default() {
                self.candle.bid
            } else {
                self.candle.ask
            };
            self.base_balance + self.quote_balance * p
        } else if currency == quote {
            let p = if self.base_balance > Default::default() {
                self.candle.ask
            } else {
                self.candle.bid
            };
            self.quote_balance + self.base_balance / p
        } else {
            assert_eq!(self.base_balance, Default::default());
            assert_eq!(self.quote_balance, Default::default());
            Default::default()
        }
    }

    /// Updates the current position based on the given trade.
    pub fn trade(&mut self, order: &mut Order) {
        order.price = match order.side {
            Side::Bid => self.candle.ask,
            Side::Ask => self.candle.bid,
        };
        let qty = order.qty
            * match order.side {
                Side::Bid => Decimal::one(),
                Side::Ask => -Decimal::one(),
            };
        self.base_balance -= qty;
        self.quote_balance += qty / order.price;
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::proptest;
    use rust_decimal::prelude::{Decimal, One};
    use std::mem;

    proptest! {
        #[test]
        fn crossing_the_spread_costs_money(symbol: super::Symbol, bid: u8, ask: u8, base: bool, id: u8, side: super::Side, qty: u8, jump: i8) {
            let q = Decimal::from(16);
            let (mut bid, mut ask, mut qty) = (Decimal::from(bid), Decimal::from(ask), Decimal::from(qty));
            bid += Decimal::one();
            ask += Decimal::one();
            qty += Decimal::one();
            if bid == ask {
                ask += Decimal::one();
            } else if bid > ask {
                mem::swap(&mut bid, &mut ask);
            }
            bid /= q;
            ask /= q;
            qty /= q;
            let candle = super::Candle{ts: super::dummy_timestamp(), symbol, bid, ask};
            let mut market = super::Market::from(candle);
            let mut order = super::Order::new(usize::from(id) + 1, candle.symbol, side, qty).unwrap().unwrap();
            let (b, q) = candle.symbol.currencies();
            let currency = if base {b} else {q};
            assert_eq!(market.pnl(currency), Default::default());
            market.trade(&mut order);
            assert!(market.pnl(currency) < Default::default());
            order.side = match order.side {
                super::Side::Bid => super::Side::Ask,
                super::Side::Ask => super::Side::Bid,
            };
            market.trade(&mut order);
            assert!(market.pnl(currency) < Default::default());
        }
    }
}
