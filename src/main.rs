#![deny(rustdoc::missing_crate_level_docs)]
#![deny(rustdoc::private_doc_tests)]
#![deny(missing_docs)]
//! FXCM Auto Trader.
use chrono::NaiveDate;
use clap::Clap;
use csv::Reader;
use reqwest::blocking::Client;
use rust_decimal::prelude::Decimal;
use std::{env, io};

mod exchange;
mod fxcm;
mod history;
mod trader;

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
    #[clap(short, long, default_value = "1000")]
    train: i16,

    /// Number of iterations to run on live exchange.
    #[clap(short, long)]
    live: i16,

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
}

fn run<E: exchange::Exchange, T: trader::Trader>(
    mut exchange: E,
    mut trader: T,
) -> fxcm::Result<Decimal> {
    while let Some(event) = exchange.next() {
        match event? {
            fxcm::Event::Candle(ref candle) => trader.on_candle(candle)?,
            fxcm::Event::Order(ref order) => trader.on_order(order)?,
        }
        while let Some(order) = trader.next() {
            exchange.insert(order?)?;
        }
    }
    exchange.pnl()
}

fn main() -> fxcm::Result<()> {
    let opts = Opts::parse();
    let (mut real, mut sim, mut dryrun, mut logging, mut history, mut reader) = Default::default();
    let mut exchange: &mut dyn exchange::Exchange = if let Ok(token) = env::var("FXCM") {
        real = Some(exchange::Real::new(opts.stage, token)?);
        real.as_mut().unwrap()
    } else {
        let rdr: &mut dyn Iterator<Item = fxcm::FallibleCandle> = if opts.replay {
            let client = Client::new();
            history = Some(history::History::new(
                move |url| Ok(client.get(url).send()?),
                opts.begin,
                opts.end,
            ));
            history.as_mut().unwrap()
        } else {
            reader = Some(
                Reader::from_reader(io::stdin())
                    .into_deserialize()
                    .map(|x| Ok(x?)),
            );
            reader.as_mut().unwrap()
        };
        sim = Some(exchange::Sim::new(opts.currency, opts.delay, rdr)?);
        sim.as_mut().unwrap()
    };
    let simulated_exchange = exchange::Sim::new(opts.currency, opts.delay, Default::default())?;
    let mut hybrid = exchange::Hybrid::new(opts.train, opts.live, simulated_exchange, exchange);
    exchange = &mut hybrid;
    if opts.noop {
        dryrun = Some(exchange::Dryrun::from(exchange));
        exchange = dryrun.as_mut().unwrap();
    }
    if opts.verbose {
        logging = Some(exchange::Logging::new(io::stdout(), exchange));
        exchange = logging.as_mut().unwrap();
    }
    Ok(println!("{}", run(exchange, trader::Dummy {})?))
}
