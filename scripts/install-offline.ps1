# Instalacja BEZ internetu — z kół przygotowanych wcześniej na maszynie z siecią
# (`python scripts\prepare_offline.py --wheels`).
#
#   .\scripts\install-offline.ps1            # środowisko .venv z vendor\wheels
#   .\scripts\install-offline.ps1 -Dev       # razem z pakietami do testów
#
# To ta sama logika co install-windows.ps1, tylko z wymuszonym trybem offline
# i pominięciem pakietów systemowych. Plik zapisany w UTF-8 z BOM — bez tego
# Windows PowerShell 5.1 czyta polskie znaki jako ANSI.

[CmdletBinding()]
param(
    [switch]$Yes,
    [switch]$Dev
)

$ErrorActionPreference = "Stop"

$arguments = @{ Offline = $true; NoSystem = $true }
if ($Yes) { $arguments.Yes = $true }
if ($Dev) { $arguments.Dev = $true }

& (Join-Path $PSScriptRoot "install-windows.ps1") @arguments
