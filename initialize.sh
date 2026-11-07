#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"
mkdir -p data

docker compose build hoerbuchgenerator
docker compose --profile initialize run --rm initialize
docker compose up -d hoerbuchgenerator

printf '%s\n' "Hoerbuchgenerator startet: http://127.0.0.1:${APP_PORT:-8000}/ui"
