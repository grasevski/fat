#!/bin/sh -eux
while getopts ':t:w:n:b:f:l:i:' opt; do
	declare $opt=$(($OPTARG+1))
done

case $t in
	1)
		stateful=false
		bidirectional=false
		;;
	2)
		stateful=false
		bidirectional=true
		;;
	*)
		stateful=true
		bidirectional=false
esac

if [ "$b" -eq 0 ]; then
	b=1
else
	b=$n
fi

cat >src/cfg.rs <<EOF
pub const STATEFUL: bool = $stateful;
pub const BIDIRECTIONAL: bool = $bidirectional;
pub const WINDOW: usize = $w;
pub const ACTIONS: usize = $n;
pub const BATCH: usize = $b;
pub const FEATURES: usize = $f;
pub const LAYERS: usize = $l;
EOF

cargo build -q --release

echo unbiased,prob,alpha,score

for u in false true; do
	for p in 0.0 0.5; do
		for a in 1e-1 1e-2 1e-3 1e-4; do
			cat <<EOF
echo "$u,$p,$a,\$(./target/release/fat run -d 1m -t 100000 -a "$a" -p "$p" -i "$i" --unbiased="$u" sim <candles.csv)"
EOF
		done
	done
done | parallel -k
