#!/usr/bin/env bash
# Download ElectricityLoadDiagrams20112014 from UCI into data/raw/
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
curl -L -o data/raw/LD2011_2014.txt.zip \
  https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip
unzip -o data/raw/LD2011_2014.txt.zip -d data/raw
