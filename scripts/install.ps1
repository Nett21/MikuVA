# Jedno wejście instalacyjne dla Windowsa.
#
#   .\scripts\install.ps1 -Full          # wszystko, z pytaniami
#   .\scripts\install.ps1 -Full -Yes     # wszystko, bez pytań
#   .\scripts\install.ps1                # sam rdzeń (jak dotąd)
#
# Ten plik niczego nie instaluje sam — przekazuje wszystkie argumenty do
# install-windows.ps1. Istnieje po to, żeby nazwa była taka sama jak na
# Linuksie i macOS-ie (scripts/install.sh): jedna rzecz do zapamiętania
# zamiast osobnej nazwy na każdy system.
#
# Plik jest zapisany w UTF-8 Z ZNACZNIKIEM BOM — bez niego Windows PowerShell 5.1
# czyta go jako ANSI i polskie znaki w komunikatach zamieniają się w krzaki.
#
# Jeśli PowerShell odmówi uruchomienia (polityka wykonywania), użyj:
#   powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1 -Full

$ErrorActionPreference = "Stop"

$target = Join-Path $PSScriptRoot "install-windows.ps1"
if (-not (Test-Path $target)) {
    Write-Host "[ERROR] Brakuje pliku $target" -ForegroundColor Red
    exit 2
}

Write-Host "[SYSTEM] Windows — uruchamiam scripts\install-windows.ps1"
& $target @args
exit $LASTEXITCODE
