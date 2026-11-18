#!/usr/bin/env sh
set -eu

status_is_ready() {
    printf '%s\n' "$1" | grep -Eq '"state"[[:space:]]*:[[:space:]]*"ready"' || return 1
    printf '%s\n' "$1" | grep -Eq '"qwen"[[:space:]]*:[[:space:]]*"[^"]+"' || return 1
    printf '%s\n' "$1" | grep -Eq '"whisper"[[:space:]]*:[[:space:]]*"[^"]+"'
}

cd "$(dirname "$0")"
mkdir -p data

if command -v curl >/dev/null 2>&1; then
    :
else
    printf '%s\n' "curl wird fuer die Bereitschaftspruefung benoetigt." >&2
    exit 1
fi

docker compose build hoerbuchgenerator
docker compose --profile initialize run --rm initialize
docker compose up -d hoerbuchgenerator

host_address=$(docker compose port hoerbuchgenerator 8000)
if [ -z "$host_address" ]; then
    printf '%s\n' "Der veroeffentlichte App-Port konnte nicht ermittelt werden." >&2
    exit 1
fi

base_url="http://${host_address}"
status_url="${base_url}/api/v1/status"
deadline=$(( $(date +%s) + 300 ))
ready=0

printf '%s' "Warte auf den vollstaendigen Modellstart (maximal 5 Minuten)"
while :; do
    remaining=$((deadline - $(date +%s)))
    if [ "$remaining" -le 0 ]; then
        break
    fi

    response=$(curl --silent --show-error --fail --max-time "$remaining" "$status_url" 2>/dev/null || :)
    if status_is_ready "$response"; then
        ready=1
        break
    fi

    remaining=$((deadline - $(date +%s)))
    if [ "$remaining" -le 0 ]; then
        break
    fi
    printf '%s' "."
    sleep_seconds=5
    if [ "$remaining" -lt "$sleep_seconds" ]; then
        sleep_seconds=$remaining
    fi
    sleep "$sleep_seconds"
done

if [ "$ready" -ne 1 ]; then
    printf '\n%s\n' "Der Container wurde gestartet, war aber innerhalb von 5 Minuten nicht bereit. Details: docker compose logs hoerbuchgenerator" >&2
    exit 1
fi

printf '\n%s\n' "Hoerbuchgenerator ist bereit:"
printf 'UI:         %s/ui\n' "$base_url"
printf 'API:        %s/docs\n' "$base_url"
printf 'API-Status: %s\n' "$status_url"
