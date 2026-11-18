$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

try {
    New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "data") | Out-Null

    if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
        throw "curl.exe wird fuer die Bereitschaftspruefung benoetigt."
    }

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

    $hostAddress = docker compose port hoerbuchgenerator 8000
    if ($LASTEXITCODE -ne 0 -or -not $hostAddress) {
        throw "Der veroeffentlichte App-Port konnte nicht ermittelt werden."
    }
    $hostAddress = ($hostAddress | Select-Object -First 1).Trim()
    $baseUrl = "http://$hostAddress"
    $statusUrl = "$baseUrl/api/v1/status"

    $timeoutSeconds = 300
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $ready = $false
    Write-Host "Warte auf den vollstaendigen Modellstart (maximal 5 Minuten)" -NoNewline
    while ($stopwatch.Elapsed.TotalSeconds -lt $timeoutSeconds) {
        $remainingSeconds = [Math]::Floor(
            $timeoutSeconds - $stopwatch.Elapsed.TotalSeconds
        )
        if ($remainingSeconds -le 0) {
            break
        }
        $response = & curl.exe `
            --silent `
            --show-error `
            --fail `
            --max-time $remainingSeconds `
            $statusUrl 2>$null
        $curlExitCode = $LASTEXITCODE
        $statusJson = $response -join "`n"

        if (
            $curlExitCode -eq 0 `
            -and $statusJson -match '"state"\s*:\s*"ready"' `
            -and $statusJson -match '"qwen"\s*:\s*"[^"]+"' `
            -and $statusJson -match '"whisper"\s*:\s*"[^"]+"'
        ) {
            $ready = $true
            break
        }

        $remainingSeconds = [Math]::Floor(
            $timeoutSeconds - $stopwatch.Elapsed.TotalSeconds
        )
        if ($remainingSeconds -le 0) {
            break
        }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds ([Math]::Min(5, $remainingSeconds))
    }

    if (-not $ready) {
        throw "Der Container wurde gestartet, war aber innerhalb von 5 Minuten nicht bereit. Details: docker compose logs hoerbuchgenerator"
    }

    Write-Output "`nHoerbuchgenerator ist bereit:"
    Write-Output "UI:         $baseUrl/ui"
    Write-Output "API:        $baseUrl/docs"
    Write-Output "API-Status: $statusUrl"
}
finally {
    Pop-Location
}
