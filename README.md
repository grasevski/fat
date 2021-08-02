# fat
FXCM Autotrader

Dependencies:
 * Rust
 * Torch bindings (see `tch-rs` project for installation instructions)
 * Hype for hyperparam tuning

Usage:
```sh
# Build debug executable
cargo build

# Build release executable
cargo build --release

# Run tests
cargo test

# Run lint
cargo clippy

# Format code
rustfmt src/*

# Show usage info
./target/release/fat -h

# Download historical minutely candle data
./target/release/fat run -nv sim -r |gzip >m1.csv.gz

# Download historical hourly candle data
./target/release/fat run -nv -f hourly sim -r |gzip >h1.csv.gz

# Download historical daily candle data
./target/release/fat run -nv -f daily sim -r |gzip >d1.csv.gz

# Backtest on historical candle data
gunzip <m1.csv.gz |head -1000000 |./target/release/fat run sim

# Backtest on historical candle data, using first 10K rows purely for training
gunzip <m1.csv.gz |head -1000000 |./target/release/fat run -t 10000 sim

# Investigate stack dump when it crashes
gunzip <m1.csv.gz |head -1000000 |RUST_BACKTRACE=full ./target/debug/fat run -t 10000 sim

# Run debugger
lldb target/debug/fat

# List available frequencies
./target/release/fat ls frequency

# List available currencies
./target/release/fat ls currency

# List available symbols
./target/release/fat ls symbol

# List available symbols for a given currency
./target/release/fat ls symbol -c AUD

# Hyperparam tuning
(time ./tune.sh) >results.csv 2>tune.log
```

Configuration of the trader (hyperparams etc) is done via command line flags as well as editing `src/cfg.rs` and recompiling.
