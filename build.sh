#!/bin/sh -eux
rustfmt --check src/*
cargo clippy --release
cargo test --release
cargo build --release
cat >score.json <<EOF
{
  "score": $(./target/release/fat run sim <d1.csv),
  "score_delay": $(./target/release/fat run -d 1d sim <d1.csv)
}
EOF
jq . <score.json
