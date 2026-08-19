"""Pluginy asystenta (Faza 11).

Każdy podkatalog to jeden plugin. Plugin dokłada narzędzia, nie zmieniając ani
jednej linii w ``brain/``, ``tools/`` czy ``main.py`` — kontrakt opisuje
:mod:`plugins.manager`.

Co jest w repozytorium:

* :mod:`plugins.reminders` — przypomnienia i timery, w całości lokalne,
* :mod:`plugins.home_assistant` — sterowanie domem przez REST API,
* :mod:`plugins.przyklad` — pusty szkielet do skopiowania.
"""

from __future__ import annotations

__all__: list[str] = []
