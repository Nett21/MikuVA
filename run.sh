#!/usr/bin/env bash
# Uruchomienie asystenta BEZ ręcznego aktywowania venv.
#
#   ./run.sh                 # okno graficzne (domyślnie)
#   ./run.sh --terminal      # rozmowa w terminalu
#   ./run.sh --check-deps    # sama diagnostyka
#
# Skrypt sam znajduje właściwego Pythona: najpierw środowisko w katalogu
# projektu (.venv), potem to, co jest w systemie. Katalog roboczy nie ma
# znaczenia — wszystko liczy się względem TEGO pliku, więc skrót na pulpicie
# albo wpis w menu aplikacji zadziała tak samo jak uruchomienie z terminala.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# MIKU_VENV_DIR pozwala trzymać środowisko poza projektem (np. na innym dysku).
VENV_DIR="${MIKU_VENV_DIR:-$PROJECT_ROOT/.venv}"

pick_python() {
    # Kolejność: środowisko projektu → python3 → python. Nie zakładamy nazwy
    # „python3": na Windowsie w Git Bashu bywa tylko „python".
    if [[ -x "$VENV_DIR/bin/python" ]]; then
        echo "$VENV_DIR/bin/python"
        return 0
    fi
    if [[ -x "$VENV_DIR/Scripts/python.exe" ]]; then  # venv utworzony na Windowsie
        echo "$VENV_DIR/Scripts/python.exe"
        return 0
    fi
    local candidate
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

if ! PYTHON="$(pick_python)"; then
    echo "[ERROR] Nie znalazłem Pythona. Zainstaluj go albo uruchom skrypt instalacyjny:" >&2
    echo "        ./scripts/install-linux-generic.sh   (albo wariant dla Twojej dystrybucji)" >&2
    exit 3
fi

if [[ ! -x "$VENV_DIR/bin/python" && ! -x "$VENV_DIR/Scripts/python.exe" ]]; then
    echo "[SYSTEM] Nie ma środowiska w $VENV_DIR — używam Pythona z systemu ($PYTHON)."
    echo "[SYSTEM] Pełna instalacja: ./scripts/install-linux-generic.sh"
fi

exec "$PYTHON" main.py "$@"
