#![deny(rustdoc::missing_crate_level_docs)]
#![deny(rustdoc::private_doc_tests)]
#![deny(missing_docs)]
//! FXCM Auto Trader.
use chrono::NaiveDate;
use csv::Reader;
use enum_map::{Enum, EnumMap};
use mimalloc::MiMalloc;
use reqwest::blocking::Client;
use rust_decimal::prelude::Decimal;
use static_assertions::const_assert;
use std::{fmt::Display, io, mem::size_of};
use structopt::StructOpt;

mod cfg;
mod exchange;
mod fxcm;
mod history;
mod model;
mod trader;

const_assert!(size_of::<trader::MrMagoo>() + size_of::<model::HiddenBatch>() <= (1 << 16));
const_assert!(size_of::<trader::MrMagoo>() + size_of::<model::ObservationBatch>() <= (1 << 16));

/// A fast cross platform allocator.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// FXCM autotrader and backtester.
#[derive(StructOpt)]
enum Opts {
    /// List available settings.
    Ls(LsCmd),

    /// Run autotrader.
    Run {
        /// Which environment to run.
        #[structopt(subcommand)]
        cmd: ExecCmd,

        /// Simulated execution delay for backtesting purposes.
        #[structopt(short, long, default_value = "1s")]
        delay: humantime::Duration,

        /// Budget in base currency.
        #[structopt(short, long, default_value = "1")]
        qty: Decimal,

        /// Base currency to be eventually settled.
        #[structopt(short, long, default_value = "USD")]
        currency: fxcm::Currency,

        /// Number of iterations to simulate.
        #[structopt(short, long, default_value = "0")]
        train: i32,

        /// Number of iterations to run on live exchange.
        #[structopt(short, long, default_value = "-1")]
        live: i32,

        /// Dont send any orders.
        #[structopt(short, long)]
        noop: bool,

        /// Log candles to stdout.
        #[structopt(short, long)]
        verbose: bool,

        /// Start date for historical data.
        #[structopt(short, long, default_value = "2012-01-01")]
        begin: NaiveDate,

        /// End date for historical data.
        #[structopt(short, long)]
        end: Option<NaiveDate>,

        /// Candle interval.
        #[structopt(short, long, default_value = "minutely")]
        when: fxcm::Frequency,

        /// Autotrader specific config.
        #[structopt(flatten)]
        cfg: trader::Cfg,
    },
}

impl Opts {
    /// Runs the backtester with the given configuration.
    fn run(self) -> fxcm::Result<()> {
        match self {
            Opts::Ls(cmd) => cmd.run(),
            Opts::Run {
                cmd,
                delay,
                qty,
                currency,
                train,
                live,
                noop,
                verbose,
                begin,
                end,
                when,
                cfg,
            } => {
                let (mut _real, mut _sim, mut _logging, mut _history, mut _reader) =
                    Default::default();
                let mut exchange: &mut dyn exchange::Exchange = match cmd {
                    ExecCmd::Real { yolo } => {
                        _real = Some(exchange::Real::new(yolo, when)?);
                        _real.as_mut().expect("real exchange not initialized")
                    }
                    ExecCmd::Sim { replay } => {
                        let rdr: &mut dyn Iterator<Item = fxcm::FallibleCandle> = if replay {
                            let client = Client::new();
                            let history = history::History::new(
                                move |url| Ok(client.get(url).send()?),
                                begin,
                                end,
                                when,
                            );
                            _history = Some(history?);
                            _history.as_mut().expect("history not initialized")
                        } else {
                            let reader = Reader::from_reader(io::stdin());
                            _reader = Some(reader.into_deserialize().map(|x| Ok(x?)));
                            _reader.as_mut().expect("reader not initialized")
                        };
                        _sim = Some(exchange::Sim::new(currency, delay, rdr)?);
                        _sim.as_mut().expect("imposssible")
                    }
                };
                let simulated_exchange = exchange::Sim::new(currency, delay, Default::default())?;
                let mut hybrid = exchange::Hybrid::new(train, live, simulated_exchange, exchange);
                exchange = &mut hybrid;
                if verbose {
                    _logging = Some(exchange::Logging::new(io::stdout(), exchange));
                    exchange = _logging.as_mut().expect("logging exchange not initialized");
                }
                let mut dryrun = trader::Dryrun::default();
                let mut mrmagoo = trader::MrMagoo::new(currency, qty, cfg)?;
                let trader: &mut dyn trader::Trader = if noop { &mut dryrun } else { &mut mrmagoo };
                println!("{}", run(exchange, trader)?);
            }
        }
        Ok(())
    }
}

/// Configuration info.
#[derive(StructOpt)]
enum LsCmd {
    /// List all available frequencies.
    Frequency,

    /// List all available currencies.
    Currency,

    /// List all available symbols.
    Symbol {
        /// Filter by currency.
        #[structopt(short, long)]
        currency: Option<fxcm::Currency>,
    },
}

impl LsCmd {
    /// Lists all enum variants.
    fn ls<T: Enum<()> + Display>() {
        for (x, _) in EnumMap::<T, ()>::default() {
            println!("{}", x);
        }
    }

    /// Runs the list command.
    fn run(self) {
        match self {
            LsCmd::Frequency => Self::ls::<fxcm::Frequency>(),
            LsCmd::Currency => Self::ls::<fxcm::Currency>(),
            LsCmd::Symbol { currency } => {
                for (symbol, _) in EnumMap::<fxcm::Symbol, ()>::default() {
                    if let Some(currency) = currency {
                        if !symbol.has_currency(currency) {
                            continue;
                        }
                    }
                    println!("{}", symbol);
                }
            }
        }
    }
}

/// Run the autotrader in the specified environment.
#[derive(StructOpt)]
enum ExecCmd {
    /// Production environment.
    Real {
        /// Whether to run against the live environment.
        #[structopt(short, long)]
        yolo: bool,
    },

    /// Run against backtester.
    Sim {
        /// Run against historical data.
        #[structopt(short, long)]
        replay: bool,
    },
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
    Opts::from_args().run()
}
