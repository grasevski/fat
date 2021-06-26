//! Trader interface and implementations.
use super::fxcm;
use enum_map::{Enum, EnumMap};
use rust_decimal::prelude::{One, ToPrimitive};
use std::convert::{TryFrom, TryInto};
use tch::{nn, nn::{Module, OptimizerConfig}, Device, Kind, Tensor};

pub type FallibleOrder = fxcm::Result<fxcm::Order>;

/// An algorithmic trading strategy.
pub trait Trader: Iterator<Item = FallibleOrder> {
    /// Runs when a candle is received.
    fn on_candle(&mut self, candle: &fxcm::Candle) -> fxcm::Result<()>;

    /// Runs when a trade is executed.
    fn on_order(&mut self, order: &fxcm::Order) -> fxcm::Result<()>;
}

/// PyTorch actor critic model.
#[derive(Debug)]
struct Model {
    /// Spectral time embedding.
    t2v: Time2Vec,

    /// RNN cell.
    mgu: Mgu,

    /// Prediction output head.
    out: nn::Linear,
}

impl Model {
    /// Initializes the model.
    fn new(p: &nn::Path, i: i8, k: i8, h: i8, o: i8) -> fxcm::Result<Self> {
        let t2v = Time2Vec::new(&(p / "t2v"), k);
        let mgu = Mgu::new(&(p / "mgu"), i + k, h);
        let out = nn::linear(&(p / "out"), h.into(), o.into(), Default::default());
        Ok(Self { t2v, mgu, out })
    }

    /// Returns the input size.
    fn in_size(&self) -> fxcm::Result<i64> {
        self.mgu.in_size()
    }

    /// Returns the non hidden output size.
    fn out_size(&self) -> fxcm::Result<i64> {
        Ok(self.out.bs.size1()?)
    }
}

impl nn::Module for Model {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let h = self.mgu.forward(&Tensor::cat(
            &[&self.t2v.forward(&xs.get(0)), xs],
            (xs.dim() - 1)
                .try_into()
                .expect("invalid model input dimension"),
        ));
        let n = (h.dim() - 1)
            .try_into()
            .expect("invalid model output dimension");
        Tensor::cat(&[self.out.forward(&h), h], n)
    }
}

/// Returns the config used for all tensors.
fn options() -> (Kind, Device) {
    (Kind::Float, Device::cuda_if_available())
}

/// Minimal Gated Unit.
#[derive(Debug)]
struct Mgu {
    /// Forget gate.
    f: nn::Linear,

    /// Hidden layer.
    h: nn::Linear,
}

impl Mgu {
    /// Initializes the MGU according to the input and hidden dimensions.
    fn new(p: &nn::Path, i: i8, h: i8) -> Self {
        let h = h.into();
        let n = i64::from(i) + h;
        let f = nn::linear(p / "f", n, h, Default::default());
        let h = nn::linear(p / "h", n, h, Default::default());
        Self { f, h }
    }

    /// Returns the input size.
    fn in_size(&self) -> fxcm::Result<i64> {
        Ok(self.f.ws.size2()?.0)
    }
}

impl nn::Module for Mgu {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let f = self.f.forward(xs).sigmoid();
        let mut t = xs.tensor_split_indices(
            &[xs.size().last().expect("xs should not be empty")
                - self.f.bs.size1().expect("f bias should be a vector")],
            (xs.dim() - 1)
                .try_into()
                .expect("invalid mgu input dimension"),
        );
        let h = self
            .h
            .forward(&Tensor::cat(
                &[&t[0], &(&f * &t[1])],
                (t[0].dim() - 1)
                    .try_into()
                    .expect("invalid mgu output dimension"),
            ))
            .tanh();
        t[1] *= 1 - &f;
        t[1] += f * h;
        t[1].shallow_clone()
    }
}

/// Fourier transform with learnable phases.
#[derive(Debug)]
struct Time2Vec(nn::Linear);

impl Time2Vec {
    /// Initializes the embedding according to the desired dimension.
    fn new(p: &nn::Path, k: i8) -> Self {
        Self(nn::linear(p / "t2v", 1, k.into(), Default::default()))
    }
}

impl nn::Module for Time2Vec {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.0.forward(xs).cos()
    }
}

/// Do nothing trader.
pub struct MrMagoo {
    /// Whether it is waiting for an order to execute.
    waiting: bool,

    /// Whether it is ready to send an order.
    ready: bool,

    /// Latest order sequence number.
    seq: usize,

    /// Remaining candle data to receive before being ready.
    remaining: i8,

    /// Settlement currency.
    currency: fxcm::Currency,

    /// PyTorch heap, may be on either GPU or CPU.
    vs: nn::VarStore,

    /// Gradient descent method.
    optimizer: nn::Optimizer<nn::AdamW>,

    /// Actor critic model.
    model: Model,

    /// All state for the trader, including hidden state.
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
        let optimizer = nn::AdamW::default().build(&vs, 1e-3)?;
        let model = Model::new(&vs.root(), i.try_into()?, 16, 16, o.try_into()?)?;
        let state = Tensor::zeros(&[model.in_size()?], options());
        Ok(Self {
            waiting: false,
            ready: false,
            seq: 0,
            remaining: n.try_into()?,
            currency,
            vs,
            optimizer,
            model,
            state,
        })
    }
}

impl Trader for MrMagoo {
    fn on_candle(&mut self, candle: &fxcm::Candle) -> fxcm::Result<()> {
        if self.remaining == EnumMap::<fxcm::Symbol, ()>::default().len().try_into()? {
            self.state = self.state.index_fill_(self.state.dim() - 1, &0.into(), candle.ts.timestamp());
        }
        let ix = 2 * i64::try_from(Enum::<()>::into_usize(candle.symbol))? + 1;
        self.state = self.state.index_fill_(self.state.dim() - 1, &ix.into(), candle.bid.to_f64().ok_or(fxcm::Error::F64(candle.bid))?);
        self.state = self.state.index_fill_(self.state.dim() - 1, &(ix + 1).into(), candle.ask.to_f64().ok_or(fxcm::Error::F64(candle.ask))?);
        self.remaining -= 1;
        if self.remaining == 0 {
            let pred = self.model.forward(&self.state);
            let t = pred.tensor_split_indices(&[self.model.out_size()?], pred.dim()? - 1);
            self.state = self.state.index_copy_(self.state.dim() - 1, , t[1]);
        }
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
