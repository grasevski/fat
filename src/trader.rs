//! Trader interface and implementations.
use super::fxcm;

pub type FallibleOrder = fxcm::Result<fxcm::Order>;

/// An algorithmic trading strategy.
pub trait Trader: Iterator<Item = FallibleOrder> {
    /// Runs when a candle is received.
    fn on_candle(&mut self, candle: &fxcm::Candle) -> fxcm::Result<()>;

    /// Runs when a trade is executed.
    fn on_order(&mut self, order: &fxcm::Order) -> fxcm::Result<()>;
}

/// Do nothing trader.
pub struct Dummy {}

impl Trader for Dummy {
    fn on_candle(&mut self, candle: &fxcm::Candle) -> fxcm::Result<()> {
        Ok(())
    }

    fn on_order(&mut self, order: &fxcm::Order) -> fxcm::Result<()> {
        Ok(())
    }
}

impl Iterator for Dummy {
    type Item = FallibleOrder;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}
