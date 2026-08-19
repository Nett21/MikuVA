"""Warstwa bezpieczeństwa: ryzyko, polityka, potwierdzenia, sandbox, audyt (Faza 7).

Wydzielona z ``tools/`` i ``brain/`` świadomie: te same reguły dotyczą narzędzi
(co deklarują), routera (co przepuszcza) i interfejsu (o co pyta użytkownika).
Trzymanie ich w którymkolwiek z tych pakietów rozmywałoby odpowiedzialność — a
przy bezpieczeństwie chodzi o to, żeby dało się przeczytać JEDEN katalog i
wiedzieć, co wolno modelowi::

    from security.policy import SecurityPolicy
    from security.confirm import default_broker
    from security.risk import RiskLevel

Ten plik jest **celowo pusty** (sam docstring), a nie zbiorem re-eksportów jak
``database/__init__.py``. Powód jest konkretny: ``security/sandbox.py`` importuje
kontrakt narzędzia z ``tools/base.py``, a ``tools/base.py`` importuje
``security/risk.py`` i ``security/confirm.py``. Gdyby import pakietu ``security``
ciągnął za sobą ``sandbox``, to ``import tools.base`` wchodziłby w cykl przez
częściowo zainicjowany moduł. Import modułów wprost usuwa problem u źródła.

Kolejność zależności: ``config`` ← ``security.risk``/``security.confirm`` ←
``tools.base`` ← ``security.sandbox``/``security.policy`` ← ``brain.tool_router``.
Nic w tym pakiecie nie importuje z ``brain/``.
"""

from __future__ import annotations

__all__ = ["audit", "confirm", "policy", "risk", "sandbox"]
