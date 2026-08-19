#!/usr/bin/env bash
# Instalacja na macOS (Homebrew).
# Cała logika jest w install-common.sh — tutaj tylko nazwy pakietów.
#
#   ./scripts/install-macos.sh            # z pytaniami
#   ./scripts/install-macos.sh --yes      # bez pytań
#   ./scripts/install-macos.sh --full     # wszystko: opcje, modele

set -euo pipefail

if command -v brew >/dev/null 2>&1; then
    PKG_LABEL="brew"
    # Homebrew instaluje do katalogu użytkownika — sudo nie jest potrzebne,
    # dlatego polecenie idzie bez podnoszenia uprawnień.
    PKG_INSTALL=(brew install)
    PKG_NEEDS_ROOT=0
    PKG_PYTHON="python"
    PKG_AUDIO="portaudio"
    # Formuła `portaudio` z Homebrew niesie i bibliotekę, i nagłówki.
    PKG_AUDIO_BUILD=""
    # ffmpeg: opcjonalny (patrz komentarz w install-apt.sh).
    PKG_FFMPEG="ffmpeg"
    # Tk dla GUI (Faza 10) — Python z Homebrew nie niesie jej sam.
    PKG_TK="python-tk"
    PKG_OLLAMA="ollama"
else
    PKG_LABEL=""
    PKG_INSTALL=()
    echo "[SYSTEM] Nie znaleziono Homebrew — pomijam pakiety systemowe."
    echo "[SYSTEM] Instalacja Homebrew: https://brew.sh"
fi

# shellcheck source=scripts/install-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/install-common.sh"
run_installer "$@"
