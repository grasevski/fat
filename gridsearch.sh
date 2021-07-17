#!/bin/bash -eux
tune () {
	cat >src/cfg.rs <<EOF
pub const LAYERS: usize = $1;
pub const FEATURES: usize = $2;
pub const SEQ_LEN: usize = $3;
pub const STEPS: usize = $4;
pub const BATCHSIZE: usize = STEPS;
pub const STATEFUL: bool = $5;
pub const HAS_BIASES: bool = $6;
pub const BIDIRECTIONAL: bool = $7;
EOF
	cargo build --release
	./target/release/fat -t 100000 -a 1e-1 -p 0.0 <candles.csv |paste <(echo $@) - >>tune_1_0.txt &
	./target/release/fat -t 100000 -a 1e-1 -p 0.5 <candles.csv |paste <(echo $@) - >>tune_1_5.txt &
	./target/release/fat -t 100000 -a 1e-2 -p 0.0 <candles.csv |paste <(echo $@) - >>tune_2_0.txt &
	./target/release/fat -t 100000 -a 1e-2 -p 0.5 <candles.csv |paste <(echo $@) - >>tune_2_5.txt &
	./target/release/fat -t 100000 -a 1e-3 -p 0.0 <candles.csv |paste <(echo $@) - >>tune_3_0.txt &
	./target/release/fat -t 100000 -a 1e-3 -p 0.5 <candles.csv |paste <(echo $@) - >>tune_3_5.txt &
	./target/release/fat -t 100000 -a 1e-4 -p 0.0 <candles.csv |paste <(echo $@) - >>tune_4_0.txt &
	./target/release/fat -t 100000 -a 1e-4 -p 0.5 <candles.csv |paste <(echo $@) - >>tune_4_5.txt &
}

gunzip <candles.csv.gz |head -1000000 >candles.csv
rm -f tune_*.txt

for layers in 1 2 4 8; do
	for features in 1 2 4 8; do
		for seq_len in 1 2 4 8; do
			for steps in 1 2 4 8; do
				for has_biases in false true; do
					tune $layers $features $seq_len $steps false $has_biases false
					tune $layers $features $seq_len $steps false $has_biases true
					tune $layers $features $seq_len $steps true $has_biases false
					wait
				done
			done
		done
	done
done
