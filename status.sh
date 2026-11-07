#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"
docker compose ps
docker compose exec -T app python -c \
    "import json,urllib.request; print(json.dumps(json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/status',timeout=5)),indent=2))"
