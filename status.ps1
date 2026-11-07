$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

try {
    docker compose ps
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    docker compose exec -T app python -c "import json,urllib.request; print(json.dumps(json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/status',timeout=5)),indent=2))"
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
