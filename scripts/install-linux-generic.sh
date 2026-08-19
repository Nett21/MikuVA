#!/usr/bin/env bash
# Instalacja na Linuksie: wariant zapasowy dla dystrybucji spoza dwóch głównych.
#
#   ./scripts/install-linux-generic.sh            # z pytaniami
#   ./scripts/install-linux-generic.sh --yes      # bez pytań
#   ./scripts/install-linux-generic.sh --dev      # razem z pakietami do testów
#   ./scripts/install-linux-generic.sh --full     # wszystko: opcje, modele
#
# Dwie rzeczy, po kolei:
#
# 1. Gdy na maszynie JEST apt albo pacman — oddajemy robotę skryptowi
#    dedykowanemu (install-apt.sh / install-pacman.sh) zamiast obsługiwać je
#    tutaj. Powód jest konkretny: tamte znają rzeczy, których ten plik nie ma
#    (nagłówki PortAudio dla apta, pakiety CUDA i opcjonalne pakiety z AUR-a dla
#    pacmana). Druga, uboższa obsługa tego samego menedżera dawałaby gorszą
#    instalację zależnie od tego, który skrypt ktoś uruchomił — a to najgorszy
#    rodzaj niespodzianki.
#
# 2. Dla pozostałych menedżerów (dnf, zypper, apk) ustawiamy nazwy pakietów tu.
#    Gdy nie ma żadnego znanego, instalator NIE ZGADUJE: przygotuje środowisko
#    Pythona i wypisze listę SKŁADNIKÓW do ręcznej instalacji. Nazwy pakietów
#    różnią się między dystrybucjami, a błędne polecenie jest gorsze niż jego brak.
#
# Menedżer wykrywamy po obecności programu w PATH, a nie po nazwie dystrybucji —
# tak samo jak robi to config.py. Lista nazw dystrybucji nigdy nie jest kompletna.

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 1. Oddaj robotę skryptowi dedykowanemu, jeśli taki istnieje ------------- #
#
# `exec` zastępuje ten proces: dalej nic się już stąd nie wykona, a kod wyjścia
# jest kodem tamtego skryptu. Bez `exec` mielibyśmy dwa procesy i podwójne
# podsumowanie.
if command -v pacman >/dev/null 2>&1; then
    echo "[SYSTEM] Wykryto pacmana — oddaję robotę scripts/install-pacman.sh"
    exec "$SCRIPTS_DIR/install-pacman.sh" "$@"
elif command -v apt-get >/dev/null 2>&1; then
    echo "[SYSTEM] Wykryto apta — oddaję robotę scripts/install-apt.sh"
    exec "$SCRIPTS_DIR/install-apt.sh" "$@"
fi

# --- 2. Pozostałe menedżery -------------------------------------------------- #

PKG_AUDIO="portaudio"
PKG_AUDIO_BUILD=""
PKG_FFMPEG="ffmpeg"
# Ollamy nie ma w repozytoriach tych dystrybucji — instalator wskaże ją linkiem.
PKG_OLLAMA=""
# Tk dla GUI (Faza 10) — nazwa pakietu różni się między dystrybucjami.
PKG_TK=""

if command -v dnf >/dev/null 2>&1; then
    PKG_LABEL="dnf"
    PKG_INSTALL=(dnf install -y)
    PKG_PYTHON="python3 python3-pip"
    PKG_TK="python3-tkinter"
    PKG_AUDIO="portaudio"
    PKG_AUDIO_BUILD="portaudio-devel"
elif command -v zypper >/dev/null 2>&1; then
    PKG_LABEL="zypper"
    PKG_INSTALL=(zypper install -y)
    PKG_PYTHON="python3 python3-pip"
    PKG_TK="python3-tk"
    PKG_AUDIO="portaudio"
    PKG_AUDIO_BUILD="portaudio-devel"
elif command -v apk >/dev/null 2>&1; then
    PKG_LABEL="apk"
    PKG_INSTALL=(apk add)
    PKG_PYTHON="python3 py3-pip"
    # Alpine nie rozdziela biblioteki i nagłówków tak jak Debian.
    PKG_AUDIO="portaudio-dev"
    PKG_TK="python3-tkinter"
else
    # Brak znanego menedżera. Pusta etykieta włącza w install-common.sh ścieżkę
    # „wypisz listę składników zamiast zgadywać nazwy pakietów".
    PKG_LABEL=""
    PKG_INSTALL=()
fi

# shellcheck source=scripts/install-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/install-common.sh"
run_installer "$@"
