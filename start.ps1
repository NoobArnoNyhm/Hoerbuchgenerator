$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

try {
    $timeoutSeconds = 600
    if ($env:STARTUP_TIMEOUT_SECONDS) {
        $timeoutSeconds = [int]$env:STARTUP_TIMEOUT_SECONDS
    }

    docker compose up -d app
    if ($LASTEXITCODE -ne 0) {
        throw "Der App-Container konnte nicht gestartet werden."
    }

    $elapsed = 0
    while ($elapsed -lt $timeoutSeconds) {
        docker compose exec -T app python -c "import json,urllib.request; s=json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/status',timeout=5)); assert s['state']=='ready' and s['models']['qwen'] and s['models']['whisper']" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $port = if ($env:APP_PORT) { $env:APP_PORT } else { "8000" }
            Write-Output "Hoerbuchgenerator bereit: http://127.0.0.1:$port/ui"
            exit 0
        }
        Start-Sleep -Seconds 5
        $elapsed += 5
    }

    docker compose logs --no-color app
    throw "Anwendung wurde innerhalb von ${timeoutSeconds}s nicht bereit."
}
finally {
    Pop-Location
}
