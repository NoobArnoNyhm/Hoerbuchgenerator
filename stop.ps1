$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

try {
    docker compose down
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
