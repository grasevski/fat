//! Trader interface and implementations.
use super::fxcm;
use enum_map::{Enum, EnumMap};
use rust_decimal::prelude::One;
use std::convert::{TryFrom, TryInto};
use tch::{nn, Device, Kind, Tensor};

pub type FallibleOrder = fxcm::Result<fxcm::Order>;

/// An algorithmic trading strategy.
pub trait Trader: Iterator<Item = FallibleOrder> {
    /// Runs when a candle is received.
    fn on_candle(&mut self, candle: &fxcm::Candle) -> fxcm::Result<()>;

    /// Runs when a trade is executed.
    fn on_order(&mut self, order: &fxcm::Order) -> fxcm::Result<()>;
}

#[derive(Debug)]
struct Model {
    t2v: Time2Vec,
    mgu: Mgu,
}

impl Model {
    fn new(path: &nn::Path, i: i8, k: i8, h: i8, o: i8) -> fxcm::Result<Self> {
        let t2v = Time2Vec::new(path, k.into());
        let mgu = Mgu::new(path, i + i8::try_from(t2v.size())?, h, o);
        Ok(Self { t2v, mgu })
    }
}

impl nn::Module for Model {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.mgu.forward(&Tensor::cat(&[xs, &self.t2v.forward(&xs.get(0))], 0))
    }
}

#[derive(Debug)]
struct Mgu {
    i: i64,
    f: nn::Linear,
    h: nn::Linear,
    o: nn::Linear,
    s: Tensor,
}

fn options() -> (Kind, Device) {
    (Kind::Float, Device::cuda_if_available())
}

impl Mgu {
    fn new(path: &nn::Path, i: i8, h: i8, o: i8) -> Self {
        let (i, h, o) = (i.into(), h.into(), o.into());
        let n = i + h;
        Self {
            i,
            f: nn::linear(path, n, h, Default::default()),
            h: nn::linear(path, n, h, Default::default()),
            o: nn::linear(path, h, o, Default::default()),
            s: Tensor::zeros(&[h], options()),
        }
    }
}

impl nn::Module for Mgu {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let f = self.f.forward(&Tensor::cat(&[xs, &self.s], 0)).sigmoid();
        let h = self.h.forward(&Tensor::cat(&[xs, &(&f * &self.s)], 0)).tanh();
        self.s *= 1 - &f;
        self.s += f * h;
        self.o.forward(&self.s)
    }
}

#[derive(Debug)]
struct Time2Vec(nn::Linear);

impl Time2Vec {
    fn new(path: &nn::Path, k: i8) -> Self {
        Self(nn::linear(path, 1, k.into(), Default::default()))
    }

    fn size(&self) -> i64 {
        self.0.bs.size()[0]
    }
}

impl nn::Module for Time2Vec {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.0.forward(xs).cos()
    }
}

/// Do nothing trader.
pub struct MrMagoo {
    waiting: bool,
    ready: bool,
    seq: usize,
    currency: fxcm::Currency,
    vs: nn::VarStore,
    model: Model,
    state: Tensor,
}

impl TryFrom<fxcm::Currency> for MrMagoo {
    type Error = fxcm::Error;

    fn try_from(currency: fxcm::Currency) -> fxcm::Result<Self> {
        let n = EnumMap::<fxcm::Symbol, ()>::default().len();
        let o = (0..n)
            .map(|x| <fxcm::Symbol as Enum<()>>::from_usize(x).currencies())
            .filter(|&(b, q)| b == currency || q == currency)
            .count()
            + 1;
        let i = 2 * n + o;
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let model = Model::new(&vs.root(), i.try_into()?, 16, 16, o.try_into()?)?;
        let state = Tensor::zeros(&[i.try_into()?], options());
        Ok(Self {
            waiting: false,
            ready: false,
            seq: 0,
            currency,
            vs,
            model,
            state,
        })
    }
}

impl Trader for MrMagoo {
    fn on_candle(&mut self, candle: &fxcm::Candle) -> fxcm::Result<()> {
        if !self.ready && candle.symbol == fxcm::Symbol::EurUsd {
            self.ready = true;
        }
        Ok(())
    }

    fn on_order(&mut self, order: &fxcm::Order) -> fxcm::Result<()> {
        self.waiting = false;
        Ok(())
    }
}

impl Iterator for MrMagoo {
    type Item = FallibleOrder;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.waiting && self.ready {
            self.waiting = true;
            self.ready = false;
            self.seq += 1;
            fxcm::Order::new(self.seq, fxcm::Symbol::EurUsd, fxcm::Side::Bid, One::one())
        } else {
            None
        }
    }
}
