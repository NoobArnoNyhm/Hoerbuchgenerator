#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"
docker compose logs --follow --tail=100 app
