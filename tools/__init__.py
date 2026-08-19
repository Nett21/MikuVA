"""Narzędzia asystenta: to, co model może zlecić do zrobienia (Faza 7).

Pakiet zawiera **kontrakt** (``base.py``), **rejestr** (``registry.py``) i same
narzędzia (``system.py``, a od Fazy 8 kolejne moduły). Model nigdy nie wywołuje
tego kodu bezpośrednio — wszystko idzie przez ``brain/tool_router.py``, który
waliduje argumenty, sprawdza politykę i pyta o zgodę::

    from tools.registry import build_registry

    registry = build_registry(settings)      # Faza 7: jedno narzędzie SAFE

Import pakietu nie uruchamia niczego zależnego od systemu: narzędzia deklarują
swoją dostępność (:meth:`tools.base.Tool.available`), a niedostępne po prostu nie
trafiają na listę pokazywaną modelowi.
"""

from __future__ import annotations

__all__ = ["base", "dependencies", "registry", "system"]
