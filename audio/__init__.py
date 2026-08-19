"""Warstwa audio: mikrofon, VAD, transkrypcja, synteza mowy i odtwarzanie.

Import tego pakietu jest lekki — biblioteki natywne (``sounddevice``,
``faster_whisper``, ``webrtcvad``, ``piper``) są ładowane dopiero w momencie
realnego użycia, żeby brak mikrofonu, głośnika albo zainstalowanego pakietu nie
blokował uruchomienia asystenta w trybie tekstowym.
"""

from __future__ import annotations

__all__ = [
    "dependencies",
    "microphone",
    "output",
    "pipeline",
    "resample",
    "tts",
    "vad",
    "whisper",
]
