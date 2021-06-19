//! Trader interface and implementations.
use super::fxcm;
use rust_decimal::prelude::One;

pub type FallibleOrder = fxcm::Result<fxcm::Order>;

/// An algorithmic trading strategy.
pub trait Trader: Iterator<Item = FallibleOrder> {
    /// Runs when a candle is received.
    fn on_candle(&mut self, candle: &fxcm::Candle) -> fxcm::Result<()>;

    /// Runs when a trade is executed.
    fn on_order(&mut self, order: &fxcm::Order) -> fxcm::Result<()>;
}

/// Do nothing trader.
#[derive(Default)]
pub struct MrMagoo {
    initialized: bool,
    ready: bool,
    seq: usize,
}

impl Trader for MrMagoo {
    fn on_candle(&mut self, candle: &fxcm::Candle) -> fxcm::Result<()> {
        if !self.initialized && candle.symbol == fxcm::Symbol::EurUsd {
            self.initialized = true;
            self.ready = true;
        }
        Ok(())
    }

    fn on_order(&mut self, order: &fxcm::Order) -> fxcm::Result<()> {
        self.ready = true;
        Ok(())
    }
}

impl Iterator for MrMagoo {
    type Item = FallibleOrder;

    fn next(&mut self) -> Option<Self::Item> {
        if self.initialized && self.ready {
            self.ready = false;
            self.seq += 1;
            fxcm::Order::new(self.seq, fxcm::Symbol::EurUsd, fxcm::Side::Ask, One::one())
        } else {
            None
        }
    }
}
