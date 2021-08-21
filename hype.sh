#!/bin/sh -eux
d=1s
while getopts ':d:t:w:n:b:f:l:i:' opt; do
	if [ "$opt" = "d" ]; then
		d=$OPTARG
	else
		declare $opt=$(($OPTARG+1))
	fi
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
echo "$u,$p,$a,\$(./target/release/fat run -d "$d" -a "$a" -p "$p" -i "$i" --unbiased="$u" sim <d1.csv)"
EOF
		done
	done
done | parallel -k
