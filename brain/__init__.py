"""Warstwa "mózgu" asystenta: model językowy, historia rozmowy, pamięć, osobowość.

Moduły tego pakietu nie dotykają systemu operacyjnego ani ścieżek — wszystko,
co platformozależne, pochodzi z ``config.py``. Zapisu na dysk też nie robią
same: ``memory.py`` woła repozytoria z pakietu ``database``.

Od Fazy 6 dochodzi pamięć semantyczna: ``embeddings.py`` (zamiana tekstu na
wektor, wyłącznie lokalnie), ``vectorstore.py`` (indeks i wyszukiwanie po
podobieństwie) oraz ``remember.py`` (polecenia „zapamiętaj/zapomnij").
"""

from __future__ import annotations

__all__ = [
    "conversation",
    "dependencies",
    "embeddings",
    "llm",
    "memory",
    "personality",
    "remember",
    "vectorstore",
]
