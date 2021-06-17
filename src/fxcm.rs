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
    pub fn new(symbol: Symbol, historical: Historical) -> self::Result<Self> {
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

/// Whether it is a buy order or a sell order.
#[derive(Clone, Copy, Debug, Serialize)]
pub enum Side {
    Bid,
    Ask,
}

/// All possible events that can be received from the exchange.
#[derive(Serialize)]
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
    const fn currencies(self) -> (Currency, Currency) {
        match self {
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
        }
    }
}

/// Tracks the current position and pnl for a given symbol.
pub struct Market {
    candle: Candle,
    position: Decimal,
}

impl From<Candle> for Market {
    fn from(candle: Candle) -> Self {
        Self {
            candle,
            position: Default::default(),
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
        let p = if self.position > Default::default() {
            self.candle.bid
        } else {
            self.candle.ask
        };
        self.position
            * if currency == base {
                p
            } else if currency == quote {
                -Decimal::one() / p
            } else {
                Default::default()
            }
    }

    /// Updates the current position based on the given trade.
    pub fn trade(&mut self, currency: Currency, order: &mut Order) -> Decimal {
        order.price = match order.side {
            Side::Bid => self.candle.ask,
            Side::Ask => self.candle.bid,
        };
        let qty = order.qty
            * match order.side {
                Side::Bid => Decimal::one(),
                Side::Ask => -Decimal::one(),
            };
        self.position += qty;
        let (base, quote) = self.candle.symbol.currencies();
        if currency == base {
            qty
        } else if currency == quote {
            -qty * order.price
        } else {
            Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::proptest;

    proptest! {
        #[test]
        fn symbols_have_different_currencies(symbol: super::Symbol) {
            let (a, b) = symbol.currencies();
            assert_ne!(a, b);
        }
    }
}
