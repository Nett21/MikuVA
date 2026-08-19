#!/usr/bin/env bash
# Instalacja BEZ internetu — z kół przygotowanych wcześniej na maszynie z siecią
# (`python scripts/prepare_offline.py --wheels`).
#
#   ./scripts/install-offline.sh            # środowisko .venv z vendor/wheels
#   ./scripts/install-offline.sh --dev      # razem z pakietami do testów
#
# To ta sama logika co pozostałe instalatory, tylko z wymuszonym trybem offline
# i pominięciem pakietów systemowych: na maszynie bez sieci menedżer pakietów
# i tak nic nie pobierze, a `--no-index` odcina pip od internetu, więc brak
# pakietu kończy się czytelnym błędem zamiast cichego pobierania.

set -euo pipefail

PKG_LABEL=""
PKG_INSTALL=()

# shellcheck source=scripts/install-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/install-common.sh"
run_installer --offline --no-system "$@"
