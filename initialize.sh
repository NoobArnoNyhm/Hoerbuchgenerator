#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"
mkdir -p data

docker compose build app
docker compose --profile initialize run --rm initialize
