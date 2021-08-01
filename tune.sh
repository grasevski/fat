#!/bin/sh -eux
gunzip <candles.csv.gz |head -1000000 >candles.csv
hype -i 100 -m "$(cat hype.json)" ./hype.sh
