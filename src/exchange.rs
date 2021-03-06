//! Exchange interface and implementations.
use super::fxcm;
use chrono::{DateTime, Duration, Utc};
use csv::Writer;
use enum_map::EnumMap;
use heapless::spsc::Queue;
use rust_decimal::prelude::Decimal;
use std::{convert::TryInto, io::Write, iter};

/// The exchange may fail to return an event.
pub type FallibleEvent = fxcm::Result<fxcm::Event>;

/// Use decorator pattern to have different exchange configurations.
pub trait Exchange: Iterator<Item = FallibleEvent> {
    fn insert(&mut self, order: fxcm::Order) -> fxcm::Result<()>;
    fn pnl(&self) -> fxcm::Result<Decimal>;
}

impl<E: Exchange + ?Sized> Exchange for &mut E {
    fn insert(&mut self, order: fxcm::Order) -> fxcm::Result<()> {
        (**self).insert(order)
    }

    fn pnl(&self) -> fxcm::Result<Decimal> {
        (**self).pnl()
    }
}

/// A connection to FXCM.
pub struct Real {
    // TODO
}

impl Real {
    /// Initializes connection to FXCM.
    pub fn new(_yolo: bool, _frequency: fxcm::Frequency) -> fxcm::Result<Self> {
        // TODO
        Ok(Self {})
    }
}

impl Exchange for Real {
    fn insert(&mut self, _: fxcm::Order) -> fxcm::Result<()> {
        // TODO
        Ok(())
    }

    fn pnl(&self) -> fxcm::Result<Decimal> {
        // TODO
        Ok(Default::default())
    }
}

impl Iterator for Real {
    type Item = FallibleEvent;

    fn next(&mut self) -> Option<Self::Item> {
        // TODO
        None
    }
}

/// An exchange with verbose logging.
pub struct Logging<W: Write, E: Exchange> {
    dst: Writer<W>,
    exchange: E,
}

impl<W: Write, E: Exchange> Logging<W, E> {
    /// Initializes logging with given exchange and output stream.
    pub fn new(wtr: W, exchange: E) -> Self {
        let dst = Writer::from_writer(wtr);
        Self { dst, exchange }
    }
}

impl<W: Write, E: Exchange> Exchange for Logging<W, E> {
    fn insert(&mut self, order: fxcm::Order) -> fxcm::Result<()> {
        self.exchange.insert(order)
    }

    fn pnl(&self) -> fxcm::Result<Decimal> {
        self.exchange.pnl()
    }
}

impl<W: Write, E: Exchange> Iterator for Logging<W, E> {
    type Item = FallibleEvent;

    fn next(&mut self) -> Option<Self::Item> {
        let ret = self.exchange.next();
        if let Some(Ok(fxcm::Event::Candle(ref r))) = ret {
            if let Err(e) = self.dst.serialize(r) {
                return Some(Err(fxcm::Error::from(e)));
            }
        }
        ret
    }
}

/// Runs training on simulated exchange, then live on actual exchange.
pub struct Hybrid<E: Exchange> {
    /// Number of iterations to run on simulated exchange, or negative to run indefinitely.
    train: i32,

    /// Number of iterations to run on live exchange, or negative to run indefinitely.
    live: i32,

    /// Simulated exchange.
    sim: Sim<iter::Empty<fxcm::FallibleCandle>>,

    /// Real exchange.
    exchange: E,
}

impl<E: Exchange> Hybrid<E> {
    /// Initializes based on number of training and live iterations.
    pub fn new(
        train: i32,
        live: i32,
        sim: Sim<iter::Empty<fxcm::FallibleCandle>>,
        exchange: E,
    ) -> Self {
        Self {
            train,
            live,
            sim,
            exchange,
        }
    }
}

impl<E: Exchange> Exchange for Hybrid<E> {
    fn insert(&mut self, order: fxcm::Order) -> fxcm::Result<()> {
        if self.train != 0 {
            self.sim.insert(order)
        } else {
            self.exchange.insert(order)
        }
    }

    fn pnl(&self) -> fxcm::Result<Decimal> {
        self.exchange.pnl()
    }
}

impl<E: Exchange> Iterator for Hybrid<E> {
    type Item = FallibleEvent;

    fn next(&mut self) -> Option<Self::Item> {
        if self.sim.needs_candle() && self.train != 0 {
            self.train -= i32::from(self.train > 0);
            if let Some(event) = self.exchange.next() {
                match event {
                    Ok(x) => match x {
                        fxcm::Event::Candle(candle) => self.sim.set_candle(candle),
                        fxcm::Event::Order(order) => {
                            return Some(Err(fxcm::Error::Order(order)));
                        }
                    },
                    Err(x) => {
                        return Some(Err(x));
                    }
                }
            }
        }
        let ret = self.sim.next();
        if ret.is_some() {
            return ret;
        }
        let ret = self.exchange.next();
        if let Some(Ok(fxcm::Event::Candle(_))) = ret {
            if self.live == 0 {
                return None;
            }
            self.live -= i32::from(self.live > 0);
        }
        ret
    }
}

/// Simulated exchange.
pub struct Sim<S: Iterator<Item = fxcm::FallibleCandle>> {
    /// Settlement currency.
    currency: fxcm::Currency,

    /// Artificial order insertion delay.
    delay: Duration,

    /// Current simulated time.
    ts: Option<DateTime<Utc>>,

    /// Most recently received candle.
    candle: Option<fxcm::Candle>,

    /// Candle data source.
    src: S,

    /// Order queue.
    orders: Queue<fxcm::Order, { fxcm::Order::MAX }>,

    /// Market data.
    markets: EnumMap<fxcm::Symbol, Option<fxcm::Market>>,
}

impl<S: Iterator<Item = fxcm::FallibleCandle>> Sim<S> {
    /// Initializes simulated exchange from config.
    pub fn new(currency: fxcm::Currency, delay: humantime::Duration, src: S) -> fxcm::Result<Self> {
        let (ts, candle, markets) = Default::default();
        Ok(Self {
            currency,
            delay: Duration::seconds(delay.as_secs().try_into()?),
            ts,
            candle,
            src,
            orders: Queue::new(),
            markets,
        })
    }

    /// Executes the order assuming no slippage.
    fn trade(&mut self, mut order: fxcm::Order) -> Option<FallibleEvent> {
        let ret = if let Some(ref mut market) = self.markets[order.symbol] {
            market.trade(&mut order);
            Ok(fxcm::Event::Order(order))
        } else {
            Err(fxcm::Error::Order(order))
        };
        Some(ret)
    }

    /// Returns Whether the simulator has a current candle.
    fn needs_candle(&self) -> bool {
        self.candle.is_none()
    }

    /// Sets the current candle from an external data source.
    fn set_candle(&mut self, candle: fxcm::Candle) {
        self.ts = Some(candle.ts);
        self.candle = Some(candle.clone());
        let symbol = candle.symbol;
        if let Some(ref mut market) = self.markets[symbol] {
            market.update(candle);
        } else {
            self.markets[symbol] = Some(candle.into());
        }
    }
}

impl<S: Iterator<Item = fxcm::FallibleCandle>> Exchange for Sim<S> {
    fn insert(&mut self, mut order: fxcm::Order) -> fxcm::Result<()> {
        if let Some(ts) = self.ts {
            order.ts = ts + self.delay;
            self.orders.enqueue(order)?;
            Ok(())
        } else {
            Err(fxcm::Error::Order(order))
        }
    }

    fn pnl(&self) -> fxcm::Result<Decimal> {
        let ret: Decimal = self
            .markets
            .values()
            .filter_map(|m| m.as_ref().map(|x| x.pnl(self.currency)))
            .sum();
        Ok(ret)
    }
}

impl<S: Iterator<Item = fxcm::FallibleCandle>> Iterator for Sim<S> {
    type Item = FallibleEvent;

    fn next(&mut self) -> Option<Self::Item> {
        if self.candle.is_none() {
            if let Some(candle) = self.src.next() {
                match candle {
                    Ok(candle) => {
                        self.set_candle(candle);
                    }
                    Err(e) => {
                        return Some(Err(e));
                    }
                }
            }
        }
        if let Some(candle) = self.candle.take() {
            if let Some(order) = self.orders.peek() {
                if order.ts < candle.ts {
                    return if let Some(order) = self.orders.dequeue() {
                        self.trade(order)
                    } else {
                        Some(Err(fxcm::Error::Initialization))
                    };
                }
            }
            Some(Ok(fxcm::Event::Candle(candle)))
        } else if let Some(order) = self.orders.dequeue() {
            self.trade(order)
        } else {
            None
        }
    }
}
