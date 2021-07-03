# fat
FXCM Autotrader

Dependencies:
 * Rust
 * Torch bindings (see `tch-rs` project for installation instructions)

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

# Download historical candle data
./target/release/fat -nrv |gzip >candles.csv.gz

# Backtest on historical candle data
gunzip <candles.csv.gz |head -1000000 |./target/release/fat

# Backtest on historical candle data, using first 10K rows purely for training
gunzip <candles.csv.gz |head -1000000 |./target/release/fat -t 10000

# Investigate stack dump when it crashes
gunzip <candles.csv.gz |head -1000000 |RUST_BACKTRACE=full ./target/debug/fat -t 10000

# Run debugger
lldb target/debug/fat
```

Configuration of the trader (hyperparams etc) is mostly done by editing the source code in `src/trader.rs` but this could be potentially configured by a file or command line arguments in the future.
