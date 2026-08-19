#!/usr/bin/env bash
# Instalacja na Arch Linux / Omarchy / Manjaro (pacman).
# Cała logika jest w install-common.sh — tutaj tylko nazwy pakietów.
#
#   ./scripts/install-pacman.sh            # z pytaniami
#   ./scripts/install-pacman.sh --yes      # bez pytań
#   ./scripts/install-pacman.sh --dev      # razem z pakietami do testów
#   ./scripts/install-pacman.sh --full     # wszystko: opcje, modele, CUDA

set -euo pipefail

PKG_LABEL="pacman"
PKG_INSTALL=(pacman -S --needed)
# Na Archu moduł venv i pip są częścią pakietu `python`.
PKG_PYTHON="python python-pip"
PKG_AUDIO="portaudio"
# Na Archu portaudio niesie i bibliotekę, i nagłówki — nie ma osobnego -dev.
PKG_AUDIO_BUILD=""
# ffmpeg: patrz komentarz w install-apt.sh. Opcjonalny.
PKG_FFMPEG="ffmpeg"
# Tk dla GUI (Faza 10).
PKG_TK="tk"
PKG_OLLAMA="ollama"
# CUDA + cuDNN dla trybu --full: bez cuDNN Whisper cofa się na CPU mimo karty.
# Świadomie BEZ `ollama-cuda`: ten pakiet zastępuje `ollama`, a podmiany
# zainstalowanego programu nie robimy przy okazji — powie o niej --check-deps.
PKG_GPU="cuda cudnn"

# --- AUR --------------------------------------------------------------------- #
# Wszystko, czego ten projekt potrzebuje, jest w repozytoriach oficjalnych
# (extra/community): python, python-pip, portaudio, tk, ffmpeg, ollama, cuda,
# cudnn. AUR jest potrzebny tylko dla rzeczy OPCJONALNYCH — dziś: program
# `piper` jako binarka systemowa (pakiet `piper-tts-bin`), gdy ktoś woli go
# zamiast pakietu Pythona.
#
# Pomocnika AUR NIE zakładamy i NIE instalujemy: `paru` i `yay` same pochodzą
# z AUR-a, więc instalowanie ich w tle byłoby budowaniem obcego kodu bez pytania.
# Gdy któryś JEST — proponujemy go użyć. Gdy nie ma — wypisujemy polecenia do
# ręcznego wykonania i idziemy dalej.
PKG_AUR_OPTIONAL="piper-tts-bin"

# shellcheck source=scripts/install-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/install-common.sh"
run_installer "$@"
