"""System pluginów: rozszerzenia bez dotykania kodu asystenta (Faza 11).

Plugin to katalog z pakietem Pythona, który wystawia obiekt ``PLUGIN`` (albo
funkcję ``create_plugin()``). Menedżer go znajduje, ładuje i pyta o narzędzia —
a te trafiają do **tego samego rejestru**, przez **ten sam router** i te same
bramki uprawnień co narzędzia wbudowane (Faza 7).

To ostatnie zdanie jest całym sensem tego pliku. Najłatwiejszym sposobem na
„system pluginów" byłoby wywołanie ich kodu wprost, z pominięciem walidacji
argumentów i potwierdzeń — i to samo byłoby jego największą dziurą: nazwa
katalogu decydowałaby o tym, czy coś wymaga zgody użytkownika. Tutaj plugin nie
ma jak niczego ominąć, bo **nie wykonuje niczego sam**: oddaje narzędzia, a
wykonaniem zajmuje się router.

Czego plugin NIE może
---------------------

* **wykonać czegokolwiek z pominięciem bramek** — router waliduje argumenty,
  liczy budżet tury, pyta o zgodę i pisze do audytu tak samo jak dla ``fs.write``,
* **zadeklarować sobie niższego ryzyka niż faktyczne** — ryzyko jest polem
  narzędzia (``ToolSpec.risk``), a eskalacja z argumentów jest jednokierunkowa
  (``BaseTool.effective_risk``); brak deklaracji znaczy CRITICAL, czyli zablokowane,
* **wywrócić asystenta** — błąd importu, błąd budowy narzędzi i błąd w
  ``poll()`` są przechwytywane i zamieniane w komunikat. Zły plugin odbiera
  użytkownikowi SIEBIE, a nie resztę programu.

Skąd biorą się pluginy
----------------------

1. z katalogu ``plugins/`` w repozytorium,
2. z katalogu wskazanego przez ``PLUGINS_EXTRA_DIR`` — miejsce na własne
   rozszerzenia użytkownika, których nie chce trzymać w repozytorium.

Nazwy filtrowane są przez ``PLUGINS_ALLOWED`` i ``PLUGINS_DISABLED``, dokładnie
tak jak nazwy narzędzi przez ``TOOLS_ALLOWED``/``TOOLS_DISABLED``.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Protocol, runtime_checkable

from config import PROJECT_ROOT, Settings, get_settings
from tools.base import Tool

logger = logging.getLogger(__name__)

# Katalog pluginów wbudowanych. Liczony od TEGO pliku, nie od katalogu roboczego:
# asystent bywa uruchamiany skrótem z menu, z innego katalogu niż projekt.
BUILTIN_PLUGINS_DIR: Final[Path] = Path(__file__).resolve().parent

# Nazwa obiektu i fabryki, których szukamy w module pluginu. Dwie drogi, bo
# prosty plugin jest stałą, a taki, który potrzebuje ustawień — funkcją.
PLUGIN_ATTRIBUTE: Final[str] = "PLUGIN"
PLUGIN_FACTORY: Final[str] = "create_plugin"

# Pliki i katalogi, które nigdy nie są pluginem.
_SKIPPED: Final[frozenset[str]] = frozenset({"__pycache__", "manager.py", "__init__.py"})


class PluginError(RuntimeError):
    """Błąd pluginu nadający się do pokazania człowiekowi."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    @property
    def user_message(self) -> str:
        return f"{self.message} ({self.hint})" if self.hint else self.message


# --------------------------------------------------------------------------- #
# Co plugin dostaje i co oddaje
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PluginInfo:
    """Wizytówka pluginu — to widzi użytkownik w ``/pluginy`` i w GUI."""

    name: str
    description: str
    version: str = "1.0"
    author: str = ""
    # Jednym zdaniem: czego plugin dotyka i czego wymaga. Nie jest ozdobą —
    # użytkownik ma prawo wiedzieć, że coś chodzi do sieci albo po dysku.
    requires: str = ""

    def describe(self) -> str:
        parts = [f"{self.name} {self.version}", self.description]
        if self.requires:
            parts.append(f"wymaga: {self.requires}")
        return " — ".join(part for part in parts if part)


@dataclass(slots=True)
class PluginContext:
    """Wszystko, co plugin dostaje od asystenta — i nic więcej.

    Świadomie nie ma tu routera ani kanału potwierdzeń: plugin nie ma jak sam
    wywołać narzędzia ani sam zapytać użytkownika o zgodę. Od tego jest router.
    """

    settings: Settings = field(default_factory=get_settings)
    # Baza z Fazy 5 (albo ``None``, gdy pamięć trwała jest wyłączona). Pluginy
    # trzymające stan mają jej używać zamiast plików obok kodu.
    database: Any | None = None
    # Zegar. Wstrzykiwany z tego samego powodu co w ``ToolContext``: test ma
    # móc przesunąć czas, a plugin od terminów bez tego jest niesprawdzalny.
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    memory: Any | None = None
    workspace: Any | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PluginNotice:
    """Coś, co plugin chce powiedzieć użytkownikowi bez pytania z jego strony.

    Tak działają przypomnienia: nikt o nie nie pyta w danej chwili, a mimo to
    muszą się odezwać. Interfejs (GUI, terminal) decyduje, jak je pokazać.
    """

    plugin: str
    text: str
    # „reminder", „warning", „info" — interfejs może to rozróżniać wizualnie.
    kind: str = "info"
    # Czy warto to POWIEDZIEĆ na głos, nie tylko wypisać.
    speak: bool = False
    data: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Plugin(Protocol):
    """Minimum, którego menedżer wymaga od pluginu."""

    @property
    def info(self) -> PluginInfo: ...

    def tools(self, ctx: PluginContext) -> Sequence[Tool[Any]]:
        """Narzędzia wystawiane modelowi. Mogą być puste."""

    def available(self, ctx: PluginContext) -> tuple[bool, str]:
        """Czy plugin ma sens na TEJ maszynie i przy TEJ konfiguracji."""

    def poll(self, ctx: PluginContext) -> Sequence[PluginNotice]:
        """Czy coś się wydarzyło od ostatniego sprawdzenia (np. minął termin)."""


class BasePlugin:
    """Wygodna baza: sensowne domyślne zachowania całego kontraktu."""

    info: PluginInfo

    def __init__(self, info: PluginInfo) -> None:
        self.info = info

    def tools(self, ctx: PluginContext) -> Sequence[Tool[Any]]:
        return ()

    def available(self, ctx: PluginContext) -> tuple[bool, str]:
        return True, ""

    def poll(self, ctx: PluginContext) -> Sequence[PluginNotice]:
        return ()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.info.name!r})"


@dataclass(slots=True)
class LoadedPlugin:
    """Wynik próby załadowania jednego pluginu — także nieudanej."""

    name: str
    info: PluginInfo | None = None
    plugin: Plugin | None = None
    tools: tuple[Tool[Any], ...] = ()
    source: Path | None = None
    # Pusty łańcuch = plugin działa. Niepusty = powód, dla którego nie działa.
    error: str = ""
    disabled_reason: str = ""

    @property
    def ok(self) -> bool:
        return self.plugin is not None and not self.error and not self.disabled_reason

    def describe(self) -> str:
        if self.error:
            return f"{self.name}: BŁĄD — {self.error}"
        if self.disabled_reason:
            return f"{self.name}: nieaktywny — {self.disabled_reason}"
        names = ", ".join(tool.spec.name for tool in self.tools) or "brak narzędzi"
        return f"{self.name}: {names}"


# --------------------------------------------------------------------------- #
# Wyszukiwanie i ładowanie
# --------------------------------------------------------------------------- #


def _allowed_names(value: str) -> set[str]:
    return {item.strip().lower() for item in value.replace(";", ",").split(",") if item.strip()}


def plugin_directories(settings: Settings | None = None) -> list[Path]:
    """Katalogi przeszukiwane w poszukiwaniu pluginów, w kolejności ważności."""
    active = settings or get_settings()
    directories = [BUILTIN_PLUGINS_DIR]

    extra = active.plugins_extra_dir.strip()
    if extra:
        # Ścieżka od użytkownika: rozwijamy „~" i zmienne, a względną liczymy od
        # katalogu projektu — tak samo jak DATABASE_PATH i FS_ALLOWED_ROOTS.
        # „~" i zmienne środowiskowe rozwijamy sami; ścieżkę względną liczymy od
        # katalogu projektu — tak samo jak DATABASE_PATH i FS_ALLOWED_ROOTS.
        expanded = Path(os.path.expandvars(extra)).expanduser()
        resolved = expanded if expanded.is_absolute() else PROJECT_ROOT / expanded
        if resolved.is_dir():
            directories.append(resolved)
        else:
            logger.warning("PLUGINS_EXTRA_DIR wskazuje na %s, którego nie ma — pomijam.", resolved)
    return directories


def discover_plugin_names(settings: Settings | None = None) -> list[tuple[str, Path]]:
    """Znajdź pluginy: pary ``(nazwa, katalog)``, bez importowania czegokolwiek.

    Pluginem jest KATALOG z ``__init__.py`` — pojedyncze pliki ``.py`` świadomie
    nie, bo plugin z reguły ma więcej niż jeden moduł, a katalog daje mu miejsce
    na testy i pliki pomocnicze.
    """
    found: list[tuple[str, Path]] = []
    seen: set[str] = set()

    for directory in plugin_directories(settings):
        try:
            entries = sorted(directory.iterdir())
        except OSError as exc:  # pragma: no cover - zależne od uprawnień
            logger.warning("Nie mogę przejrzeć katalogu pluginów %s: %s", directory, exc)
            continue

        for entry in entries:
            name = entry.name
            if name in _SKIPPED or name.startswith((".", "_")):
                continue
            if not entry.is_dir() or not (entry / "__init__.py").is_file():
                continue
            if name in seen:
                logger.info("Plugin %r już znaleziony wcześniej — pomijam %s", name, entry)
                continue
            seen.add(name)
            found.append((name, entry))
    return found


def _import_plugin_module(name: str, directory: Path) -> ModuleType:
    """Zaimportuj pakiet pluginu — z repozytorium albo z katalogu użytkownika."""
    if directory.parent == BUILTIN_PLUGINS_DIR:
        return importlib.import_module(f"plugins.{name}")

    # Plugin spoza repozytorium: ładujemy go po ścieżce, bez dopisywania
    # czegokolwiek na stałe do ``sys.path``. Nazwa modułu dostaje przedrostek,
    # żeby cudzy plugin nie przesłonił modułu asystenta.
    module_name = f"miku_plugin_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(
        module_name, directory / "__init__.py", submodule_search_locations=[str(directory)]
    )
    if spec is None or spec.loader is None:
        raise PluginError(f"nie umiem zaimportować pluginu z {directory}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _plugin_from_module(module: ModuleType, name: str) -> Plugin:
    """Wyłuskaj obiekt pluginu z modułu: ``PLUGIN`` albo ``create_plugin()``."""
    candidate = getattr(module, PLUGIN_ATTRIBUTE, None)
    if candidate is None:
        factory = getattr(module, PLUGIN_FACTORY, None)
        if factory is None:
            raise PluginError(
                f"plugin {name!r} nie wystawia ani {PLUGIN_ATTRIBUTE}, ani {PLUGIN_FACTORY}()",
                hint="patrz plugins/przyklad/__init__.py",
            )
        candidate = factory()

    if not isinstance(candidate, Plugin):
        raise PluginError(
            f"plugin {name!r} nie spełnia kontraktu (brakuje info/tools/available/poll)",
            hint="najprościej odziedziczyć po plugins.manager.BasePlugin",
        )
    return candidate


class PluginManager:
    """Ładuje pluginy i oddaje ich narzędzia. Nigdy nie rzuca wyjątkiem.

    Jeden obiekt na uruchomienie asystenta. Trzyma wynik ładowania, żeby
    ``/pluginy`` i raport zależności mogły pokazać także te, które NIE działają —
    „nie ma go na liście" jest gorszą odpowiedzią niż „jest, ale brakuje tokenu".
    """

    def __init__(
        self, settings: Settings | None = None, *, context: PluginContext | None = None
    ) -> None:
        self._settings = settings or get_settings()
        self._context = context or PluginContext(settings=self._settings)
        self._loaded: list[LoadedPlugin] = []
        self._done = False

    # --- stan ------------------------------------------------------------- #

    @property
    def context(self) -> PluginContext:
        return self._context

    @property
    def loaded(self) -> Sequence[LoadedPlugin]:
        return tuple(self._loaded)

    @property
    def active(self) -> Sequence[LoadedPlugin]:
        return tuple(item for item in self._loaded if item.ok)

    def describe(self) -> str:
        """Jedna linijka do raportu zależności."""
        if not self._settings.plugins_enabled:
            return "pluginy wyłączone (PLUGINS_ENABLED=false)"
        if not self._loaded:
            return "brak pluginów"
        working = [item for item in self._loaded if item.ok]
        broken = [item for item in self._loaded if item.error]
        parts = [f"{len(working)} z {len(self._loaded)}"]
        if working:
            parts.append(", ".join(item.name for item in working))
        if broken:
            parts.append(f"z błędem: {', '.join(item.name for item in broken)}")
        return "; ".join(parts)

    # --- ładowanie -------------------------------------------------------- #

    def load(self, *, force: bool = False) -> Sequence[LoadedPlugin]:
        """Załaduj wszystkie pluginy. Powtórne wywołanie nic nie robi."""
        if self._done and not force:
            return self.loaded

        self._loaded = []
        self._done = True
        if not self._settings.plugins_enabled:
            logger.info("Pluginy wyłączone (PLUGINS_ENABLED=false).")
            return self.loaded

        allowed = _allowed_names(self._settings.plugins_allowed)
        disabled = _allowed_names(self._settings.plugins_disabled)
        wildcard = "*" in allowed or not allowed

        for name, directory in discover_plugin_names(self._settings):
            lowered = name.lower()
            if lowered in disabled:
                self._loaded.append(
                    LoadedPlugin(
                        name=name,
                        source=directory,
                        disabled_reason="wyłączony w PLUGINS_DISABLED",
                    )
                )
                continue
            if not wildcard and lowered not in allowed:
                self._loaded.append(
                    LoadedPlugin(
                        name=name,
                        source=directory,
                        disabled_reason="spoza listy PLUGINS_ALLOWED",
                    )
                )
                continue
            self._loaded.append(self._load_one(name, directory))

        return self.loaded

    def _load_one(self, name: str, directory: Path) -> LoadedPlugin:
        """Załaduj JEDEN plugin. Każdy błąd zostaje w jego wyniku, nie leci wyżej."""
        entry = LoadedPlugin(name=name, source=directory)
        try:
            module = _import_plugin_module(name, directory)
            plugin = _plugin_from_module(module, name)
        except PluginError as exc:
            entry.error = exc.user_message
            logger.warning("Plugin %s: %s", name, entry.error)
            return entry
        except Exception as exc:  # import cudzego kodu — może rzucić czymkolwiek
            entry.error = f"nie udało się załadować ({exc})"
            logger.warning("Plugin %s nie załadował się: %s", name, exc, exc_info=True)
            return entry

        entry.plugin = plugin
        try:
            entry.info = plugin.info
        except Exception as exc:  # pragma: no cover - błędna implementacja
            entry.error = f"nie umie się przedstawić ({exc})"
            return entry

        try:
            usable, reason = plugin.available(self._context)
        except Exception as exc:
            entry.error = f"sprawdzenie dostępności nie powiodło się ({exc})"
            return entry
        if not usable:
            entry.disabled_reason = reason or "niedostępny na tej maszynie"
            return entry

        try:
            entry.tools = tuple(plugin.tools(self._context))
        except Exception as exc:
            entry.error = f"nie zbudował narzędzi ({exc})"
            logger.warning("Plugin %s nie zbudował narzędzi: %s", name, exc, exc_info=True)
            return entry
        return entry

    # --- to, po co to wszystko -------------------------------------------- #

    def tools(self) -> list[Tool[Any]]:
        """Narzędzia wszystkich działających pluginów — do rejestru z Fazy 7."""
        self.load()
        collected: list[Tool[Any]] = []
        for item in self.active:
            collected.extend(item.tools)
        return collected

    def poll(self) -> list[PluginNotice]:
        """Zapytaj pluginy, czy coś się wydarzyło. Błąd jednego nie rusza reszty."""
        self.load()
        notices: list[PluginNotice] = []
        for item in self.active:
            if item.plugin is None:
                continue
            try:
                notices.extend(item.plugin.poll(self._context))
            except Exception as exc:
                logger.warning("Plugin %s zgłosił błąd przy sprawdzaniu: %s", item.name, exc)
        return notices


def load_plugin_tools(
    settings: Settings | None = None, *, context: PluginContext | None = None
) -> list[Tool[Any]]:
    """Skrót dla rejestru: same narzędzia, bez trzymania menedżera."""
    return PluginManager(settings, context=context).tools()


def iter_plugin_tools(plugins: Iterable[LoadedPlugin]) -> Iterable[Tool[Any]]:
    for item in plugins:
        if item.ok:
            yield from item.tools


__all__ = [
    "BUILTIN_PLUGINS_DIR",
    "BasePlugin",
    "LoadedPlugin",
    "Plugin",
    "PluginContext",
    "PluginError",
    "PluginInfo",
    "PluginManager",
    "PluginNotice",
    "discover_plugin_names",
    "iter_plugin_tools",
    "load_plugin_tools",
    "plugin_directories",
]
