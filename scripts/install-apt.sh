#!/usr/bin/env bash
# Instalacja na Debianie, Ubuntu, Linux Mint i pochodnych (apt).
# Cała logika jest w install-common.sh — tutaj tylko nazwy pakietów.
#
#   ./scripts/install-apt.sh            # z pytaniami
#   ./scripts/install-apt.sh --yes      # bez pytań
#   ./scripts/install-apt.sh --dev      # razem z pakietami do testów
#   ./scripts/install-apt.sh --full     # wszystko: opcje, modele

set -euo pipefail

PKG_LABEL="apt"
PKG_INSTALL=(apt-get install -y)
# python3-venv jest osobnym pakietem — bez niego `python3 -m venv` nie działa,
# co jest najczęstszą przyczyną nieudanej instalacji na Debianie/Ubuntu.
PKG_PYTHON="python3 python3-venv python3-pip"
# libportaudio2 to biblioteka wykonawcza pod `sounddevice`. Nagłówków
# (portaudio19-dev) NIE dokładamy domyślnie: koło `sounddevice` z PyPI niesie
# własną kopię PortAudio i kompilacja nie jest potrzebna. Są w PKG_AUDIO_BUILD
# na wypadek dystrybucji, dla której koła nie ma.
PKG_AUDIO="libportaudio2"
PKG_AUDIO_BUILD="portaudio19-dev"
# ffmpeg — narzędzie do konwersji dźwięku. Ani Whisper, ani Piper nie wołają go
# w tym projekcie (do modelu idzie tablica NumPy prosto z mikrofonu), ale jest
# domyślnym dekoderem plików audio dla bibliotek warstwy dźwiękowej i przydaje
# się przy pracy z nagraniami z dysku. Instalowany jako OPCJONALNY — jego brak
# nie psuje niczego, co asystent robi dzisiaj.
PKG_FFMPEG="ffmpeg"
# Debian i Ubuntu nie mają Ollamy w repozytoriach — instalator ją wskaże.
# Tk dla GUI (Faza 10) — na Debianie osobny pakiet.
PKG_TK="python3-tk"
PKG_OLLAMA=""

# shellcheck source=scripts/install-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/install-common.sh"
run_installer "$@"
