#![deny(rustdoc::missing_crate_level_docs)]
#![deny(rustdoc::private_doc_tests)]
#![deny(missing_docs)]
//! FXCM Auto Trader.
use chrono::NaiveDate;
use clap::Clap;
use csv::Reader;
use mimalloc::MiMalloc;
use reqwest::blocking::Client;
use rust_decimal::prelude::Decimal;
use std::{env, io};

mod cfg;
mod exchange;
mod fxcm;
mod history;
mod model;
mod trader;

/// A fast cross platform allocator.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// FXCM autotrader and backtester.
#[derive(Clap)]
struct Opts {
    /// Simulated execution delay for backtesting purposes.
    #[clap(short, long, default_value = "1s")]
    delay: humantime::Duration,

    /// Budget for each market.
    #[clap(short, long, default_value = "1")]
    qty: Decimal,

    /// Base currency to be eventually settled.
    #[clap(short, long, default_value = "USD")]
    currency: fxcm::Currency,

    /// Number of iterations to simulate.
    #[clap(short, long, default_value = "0")]
    train: i32,

    /// Number of iterations to run on live exchange.
    #[clap(short, long, default_value = "-1")]
    live: i32,

    /// Dont send any orders.
    #[clap(short, long)]
    noop: bool,

    /// Log candles to stdout.
    #[clap(short, long)]
    verbose: bool,

    /// Run against staging environment.
    #[clap(short, long)]
    stage: bool,

    /// Run against historical data.
    #[clap(short, long)]
    replay: bool,

    /// Start date for historical data.
    #[clap(short, long, default_value = "2012-01-01")]
    begin: NaiveDate,

    /// End date for historical data.
    #[clap(short, long)]
    end: Option<NaiveDate>,

    /// Random number generator seed.
    #[clap(short, long, default_value = "0")]
    gen: i64,

    /// Dropout rate.
    #[clap(short, long, default_value = "0")]
    prob: f64,

    /// Learning rate.
    #[clap(short, long, default_value = "1e-3")]
    alpha: f64,
}

/// Connects trader to exchange and runs to completion.
fn run(
    exchange: &mut dyn exchange::Exchange,
    trader: &mut dyn trader::Trader,
) -> fxcm::Result<Decimal> {
    while let Some(event) = exchange.next() {
        match event? {
            fxcm::Event::Candle(candle) => {
                for order in trader.on_candle(candle)? {
                    exchange.insert(order)?;
                }
            }
            fxcm::Event::Order(order) => trader.on_order(order)?,
        }
    }
    exchange.pnl()
}

/// Configures and runs the backtester and or exchange.
fn main() -> fxcm::Result<()> {
    let opts = Opts::parse();
    let (mut _real, mut _sim, mut _logging, mut _history, mut _reader) = Default::default();
    let mut exchange: &mut dyn exchange::Exchange = if let Ok(token) = env::var("FXCM") {
        _real = Some(exchange::Real::new(opts.stage, token)?);
        _real.as_mut().expect("real exchange not initialized")
    } else {
        let rdr: &mut dyn Iterator<Item = fxcm::FallibleCandle> = if opts.replay {
            let client = Client::new();
            let history =
                history::History::new(move |url| Ok(client.get(url).send()?), opts.begin, opts.end);
            _history = Some(history?);
            _history.as_mut().expect("history not initialized")
        } else {
            let reader = Reader::from_reader(io::stdin());
            _reader = Some(reader.into_deserialize().map(|x| Ok(x?)));
            _reader.as_mut().expect("reader not initialized")
        };
        _sim = Some(exchange::Sim::new(opts.currency, opts.delay, rdr)?);
        _sim.as_mut().expect("imposssible")
    };
    let simulated_exchange = exchange::Sim::new(opts.currency, opts.delay, Default::default())?;
    let mut hybrid = exchange::Hybrid::new(opts.train, opts.live, simulated_exchange, exchange);
    exchange = &mut hybrid;
    if opts.verbose {
        _logging = Some(exchange::Logging::new(io::stdout(), exchange));
        exchange = _logging.as_mut().expect("logging exchange not initialized");
    }
    let mut dryrun = trader::Dryrun::default();
    let mut mrmagoo = trader::MrMagoo::new(
        opts.currency,
        opts.qty,
        opts.train,
        opts.gen,
        opts.prob,
        opts.alpha,
        true,
    )?;
    let trader: &mut dyn trader::Trader = if opts.noop { &mut dryrun } else { &mut mrmagoo };
    println!("{}", run(exchange, trader)?);
    Ok(())
}
