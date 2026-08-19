# Uruchomienie asystenta BEZ ręcznego aktywowania venv (Windows, PowerShell).
#
#   .\run.ps1                # okno graficzne (domyślnie)
#   .\run.ps1 --terminal     # rozmowa w terminalu
#
# Skrypt sam znajduje Pythona: najpierw środowisko w katalogu projektu (.venv),
# potem to, co jest w systemie. Ścieżki liczą się względem TEGO pliku, więc skrót
# na pulpicie zadziała tak samo jak uruchomienie z konsoli.

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvDir = if ($env:MIKU_VENV_DIR) { $env:MIKU_VENV_DIR } else { Join-Path $ProjectRoot ".venv" }

$Candidates = @(
    (Join-Path $VenvDir "Scripts\python.exe"),
    (Join-Path $VenvDir "bin\python")
)
$Python = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $Python) {
    foreach ($name in @("python", "python3", "py")) {
        $found = Get-Command $name -ErrorAction SilentlyContinue
        if ($found) { $Python = $found.Source; break }
    }
    if ($Python) {
        Write-Host "[SYSTEM] Nie ma środowiska w $VenvDir - uzywam Pythona z systemu ($Python)."
        Write-Host "[SYSTEM] Pelna instalacja: .\scripts\install-windows.ps1"
    }
}

if (-not $Python) {
    Write-Error "[ERROR] Nie znalazlem Pythona. Zainstaluj go z python.org (z opcja 'tcl/tk and IDLE') albo uruchom .\scripts\install-windows.ps1"
    exit 3
}

& $Python main.py @args
exit $LASTEXITCODE
