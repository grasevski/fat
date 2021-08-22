#!/bin/sh -eux
cargo build --release
cargo clippy --release
cargo test --release
cat >score.json <<EOF
{
  "score": $(./target/release/fat run sim <d1.csv),
  "score_delay": $(./target/release/fat run -d 1d sim <d1.csv)
}
EOF
jq . <score.json
