#![deny(rustdoc::missing_crate_level_docs)]
#![deny(rustdoc::private_doc_tests)]
#![deny(missing_docs)]
//! FXCM Auto Trader.
use chrono::NaiveDate;
use clap::Clap;
use csv::Reader;
use enum_map::EnumMap;
use mimalloc::MiMalloc;
use reqwest::blocking::Client;
use rust_decimal::prelude::Decimal;
use static_assertions::const_assert;
use std::{io, mem::size_of};

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
#[derive(Clap)]
enum Opts {
    /// List available settings.
    Ls(LsCmd),

    /// Run autotrader.
    Run {
        /// Which environment to run.
        #[clap(subcommand)]
        cmd: ExecCmd,

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

        /// Start date for historical data.
        #[clap(short, long, default_value = "2012-01-01")]
        begin: NaiveDate,

        /// End date for historical data.
        #[clap(short, long)]
        end: Option<NaiveDate>,

        /// Autotrader specific config.
        #[clap(flatten)]
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
                cfg,
            } => {
                let (mut _real, mut _sim, mut _logging, mut _history, mut _reader) =
                    Default::default();
                let mut exchange: &mut dyn exchange::Exchange = match cmd {
                    ExecCmd::Real { yolo } => {
                        _real = Some(exchange::Real::new(yolo)?);
                        _real.as_mut().expect("real exchange not initialized")
                    }
                    ExecCmd::Sim { replay } => {
                        let rdr: &mut dyn Iterator<Item = fxcm::FallibleCandle> = if replay {
                            let client = Client::new();
                            let history = history::History::new(
                                move |url| Ok(client.get(url).send()?),
                                begin,
                                end,
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
#[derive(Clap)]
enum LsCmd {
    /// List all available currencies.
    Currency,

    /// List all available symbols.
    Symbol {
        /// Filter by currency.
        #[clap(short, long)]
        currency: Option<fxcm::Currency>,
    },
}

impl LsCmd {
    /// Runs the list command.
    fn run(self) {
        match self {
            LsCmd::Currency => {
                let currencies: EnumMap<fxcm::Currency, ()> = Default::default();
                for (currency, _) in currencies {
                    println!("{}", currency);
                }
            }
            LsCmd::Symbol { currency } => {
                let symbols: EnumMap<fxcm::Symbol, ()> = Default::default();
                for (symbol, _) in symbols {
                    if let Some(currency) = currency {
                        if symbol.has_currency(currency) {
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
#[derive(Clap)]
enum ExecCmd {
    /// Production environment.
    Real {
        /// Whether to run against the live environment.
        #[clap(short, long)]
        yolo: bool,
    },

    /// Run against backtester.
    Sim {
        /// Run against historical data.
        #[clap(short, long)]
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
    Opts::parse().run()
}
