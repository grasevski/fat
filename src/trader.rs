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
    out: nn::Linear,
}

impl Model {
    fn new(path: &nn::Path, i: i8, k: i8, h: i8, o: i8) -> fxcm::Result<Self> {
        let t2v = Time2Vec::new(path, k);
        let mgu = Mgu::new(path, i + k, h);
        let out = nn::linear(path, h.into(), o.into(), Default::default());
        Ok(Self { t2v, mgu, out })
    }
}

impl nn::Module for Model {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let h = self
            .mgu
            .forward(&Tensor::cat(&[&self.t2v.forward(&xs.get(0)), xs], 0));
        Tensor::cat(&[self.out.forward(&h), h], 0)
    }
}

#[derive(Debug)]
struct Mgu {
    f: nn::Linear,
    h: nn::Linear,
}

fn options() -> (Kind, Device) {
    (Kind::Float, Device::cuda_if_available())
}

impl Mgu {
    fn new(path: &nn::Path, i: i8, h: i8) -> Self {
        let h = h.into();
        let n = i64::from(i) + h;
        let f = nn::linear(path, n, h, Default::default());
        let h = nn::linear(path, n, h, Default::default());
        Self { f, h }
    }
}

impl nn::Module for Mgu {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let f = self.f.forward(xs).sigmoid();
        let mut t = xs.tensor_split1(&[xs.size1().unwrap() - self.f.bs.size1().unwrap()], 0);
        let h = self
            .h
            .forward(&Tensor::cat(&[&t[0], &(&f * &t[1])], 0))
            .tanh();
        t[1] *= 1 - &f;
        t[1] += f * h;
        t[1].shallow_clone()
    }
}

#[derive(Debug)]
struct Time2Vec(nn::Linear);

impl Time2Vec {
    fn new(path: &nn::Path, k: i8) -> Self {
        Self(nn::linear(path, 1, k.into(), Default::default()))
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
