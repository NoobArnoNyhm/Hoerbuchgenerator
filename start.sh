#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"
timeout_seconds="${STARTUP_TIMEOUT_SECONDS:-600}"

docker compose up -d app

elapsed=0
while [ "$elapsed" -lt "$timeout_seconds" ]; do
    if docker compose exec -T app python -c \
        "import json,urllib.request; s=json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/status',timeout=5)); assert s['state']=='ready' and s['models']['qwen'] and s['models']['whisper']" \
        2>/dev/null
    then
        printf '%s\n' "Hoerbuchgenerator bereit: http://127.0.0.1:${APP_PORT:-8000}/ui"
        exit 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done

printf '%s\n' "Anwendung wurde innerhalb von ${timeout_seconds}s nicht bereit." >&2
docker compose logs --no-color app
exit 1
