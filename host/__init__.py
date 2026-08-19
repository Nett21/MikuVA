"""Warstwa systemu operacyjnego: jedyne miejsce, które wie, jak działa TA maszyna.

Do Fazy 7 cała wiedza o platformie mieszkała w ``config.py``. Faza 8 dokłada
narzędzia, które naprawdę dotykają systemu (pliki, aplikacje, procesy, powłoka,
usługi) — a tego nie da się napisać bez kodu zależnego od systemu. Zamiast
rozlać go po ``tools/``, jest tutaj, w jednym pakiecie:

======================= ====================================================
moduł                   za co odpowiada
======================= ====================================================
``host/paths.py``       dozwolone katalogi, kanonizacja ścieżek, limity
``host/privileges.py``  czy działamy jako root/administrator (i odmowa)
``host/apps.py``        lista i uruchamianie aplikacji, otwieranie adresów
``host/processes.py``   lista procesów i ich zamykanie
``host/shell.py``       uruchamianie programów: argv, env, blokady
``host/services.py``    usługi użytkownika (``systemctl --user``)
======================= ====================================================

Reguły, które ten pakiet trzyma:

* **O systemie pyta ``config.detect_platform()``**, a nie ``sys.platform`` —
  detekcja jest jedna dla całego projektu (``config.py``), tutaj jest tylko jej
  UŻYCIE. Ścieżki systemowe budujemy przez ``config.path_from_env`` i
  ``config.user_data_directories``, żeby nie zgadywać układu katalogów.
* **Brak czegoś to poprawny stan, nie awaria.** Każda funkcja umie powiedzieć
  „na tej maszynie tego nie ma" (brak sesji graficznej, brak ``psutil``, brak
  ``systemctl``), a narzędzie z ``tools/`` zamienia to na „niedostępne" —
  wtedy model w ogóle go nie widzi.
* **Nic tutaj nie decyduje o uprawnieniach.** Poziomy ryzyka i potwierdzenia są
  w ``security/``, a bramki w ``brain/tool_router.py``. Ten pakiet wykonuje.

Nazwa pakietu to ``host``, a nie ``platform``, choć ARCHITECTURE.md proponuje to
drugie: pakiet ``platform`` w katalogu projektu **przesłoniłby moduł ``platform``
z biblioteki standardowej** (katalog projektu jest pierwszy na ``sys.path``), a
z niego korzysta ``config.py`` do detekcji systemu. Byłby to błąd trudny do
zdiagnozowania i zależny od katalogu uruchomienia.

Import pakietu nic nie uruchamia — importuj moduły wprost::

    from host.paths import Workspace, resolve_within
    from host.shell import run_command
"""

from __future__ import annotations

__all__ = ["apps", "paths", "privileges", "processes", "services", "shell"]
