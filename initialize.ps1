$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

try {
    New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "data") | Out-Null

    docker compose build app
    if ($LASTEXITCODE -ne 0) {
        throw "Das Runtime-Image konnte nicht gebaut werden."
    }

    docker compose --profile initialize run --rm initialize
    if ($LASTEXITCODE -ne 0) {
        throw "Die Modellinitialisierung ist fehlgeschlagen."
    }
}
finally {
    Pop-Location
}
