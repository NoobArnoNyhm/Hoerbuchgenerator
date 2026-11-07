$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

try {
    New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "data") | Out-Null

    docker compose build hoerbuchgenerator
    if ($LASTEXITCODE -ne 0) {
        throw "Das Runtime-Image konnte nicht gebaut werden."
    }

    docker compose --profile initialize run --rm initialize
    if ($LASTEXITCODE -ne 0) {
        throw "Die Modellinitialisierung ist fehlgeschlagen."
    }

    docker compose up -d hoerbuchgenerator
    if ($LASTEXITCODE -ne 0) {
        throw "Der App-Container konnte nicht gestartet werden."
    }

    $port = if ($env:APP_PORT) { $env:APP_PORT } else { "8000" }
    Write-Output "Hoerbuchgenerator startet: http://127.0.0.1:$port/ui"
}
finally {
    Pop-Location
}
