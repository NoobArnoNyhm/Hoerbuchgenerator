$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

try {
    docker compose logs --follow --tail=100 app
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
