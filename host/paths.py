"""Dozwolone katalogi i kanonizacja ścieżek (Faza 8).

To najważniejszy plik warstwy narzędzi plikowych. Zasada jest jedna:

    **Narzędzie plikowe widzi wyłącznie to, co leży w skonfigurowanych
    katalogach.** Reszta dysku nie istnieje.

Domyślnie skonfigurowany jest DOKŁADNIE JEDEN katalog: ``workspace/`` w katalogu
danych asystenta. Nie ``~``, nie ``Dokumenty``, nie dysk. Kto chce dać dostęp do
czegoś więcej, wpisuje to wprost w ``FS_ALLOWED_ROOTS`` — świadomie, raz.

Jak wygląda sprawdzenie ścieżki (:func:`resolve_within`):

1. rozwinięcie ``~`` i zmiennych środowiskowych — model może przysłać ``~/plik``,
2. ścieżka względna liczy się od PIERWSZEGO dozwolonego katalogu, nigdy od
   katalogu roboczego procesu (ten zależy od tego, skąd program uruchomiono),
3. ``resolve()`` — usuwa ``..``, ``.`` i **podąża za dowiązaniami symbolicznymi**,
4. dopiero na wyniku sprawdzamy zawieranie w dozwolonym katalogu.

Kolejność 3 → 4 jest tu całą ochroną: dowiązanie ``workspace/skrót`` wskazujące
na ``/etc`` po ``resolve()`` JEST ``/etc``, więc wypada z dozwolonego obszaru.
Sprawdzanie przed rozwinięciem dałoby ochronę pozorną.

Porównanie wielkości litery zależy od systemu plików, nie od naszego gustu:
na Windowsie i macOS ``C:\\Dane`` i ``c:\\dane`` to ten sam katalog, na Linuksie
dwa różne. Bierzemy to z detekcji platformy — inaczej ten sam config działałby
inaczej na dwóch maszynach.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from config import (
    OSFamily,
    PlatformInfo,
    Settings,
    app_data_directory,
    detect_platform,
    get_settings,
    path_from_env,
)

logger = logging.getLogger(__name__)

# Nazwa katalogu roboczego asystenta — jedynego domyślnie dozwolonego.
WORKSPACE_DIR_NAME: Final[str] = "workspace"

# Ile bajtów oglądamy, szukając bajtu zerowego (plik binarny).
_BINARY_PROBE_BYTES: Final[int] = 8_192

# Rozdzielniki listy katalogów w ``.env``. Średnik i przecinek, ale NIE
# ``os.pathsep``: ten na Windowsie jest średnikiem, a na Uniksie dwukropkiem —
# a dwukropek jest częścią ścieżki windowsowej (``C:\\dane``). Jeden zapis listy
# dla wszystkich systemów jest mniej zaskakujący niż „to zależy".
_LIST_SEPARATORS: Final[str] = ";,"


class PathNotAllowedError(Exception):
    """Ścieżka odrzucona — wraca do modelu jako czytelny błąd, nie jako awaria."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _split_roots(raw: str) -> list[str]:
    items = [raw]
    for separator in _LIST_SEPARATORS:
        items = [part for item in items for part in item.split(separator)]
    return [item.strip() for item in items if item.strip()]


def default_workspace(settings: Settings | None = None) -> Path:
    """Katalog roboczy asystenta — domyślnie jedyny dozwolony.

    Leży w katalogu danych programu (tam, gdzie baza), a nie w katalogu projektu:
    projekt bywa tylko do odczytu, a dane użytkownika mają przetrwać
    przeinstalowanie.
    """
    return app_data_directory() / WORKSPACE_DIR_NAME


def configured_roots(settings: Settings | None = None) -> list[Path]:
    """Dozwolone katalogi z konfiguracji (jeszcze bez sprawdzania, czy istnieją).

    Pusty ``FS_ALLOWED_ROOTS`` znaczy „tylko katalog roboczy asystenta". To celowo
    zachowawcze: narzędzie plikowe na świeżej instalacji nie ma dostępu do żadnego
    prywatnego pliku, dopóki użytkownik sam tego nie zmieni.
    """
    active = settings or get_settings()
    roots: list[Path] = []
    for raw in _split_roots(active.fs_allowed_roots):
        expanded = Path(os.path.expandvars(raw)).expanduser()
        if not expanded.is_absolute():
            logger.warning(
                "FS_ALLOWED_ROOTS: pomijam %r — dozwolone katalogi muszą być "
                "podane ścieżką bezwzględną.",
                raw,
            )
            continue
        roots.append(expanded)
    if not roots:
        roots.append(default_workspace(active))
    return roots


@dataclass(frozen=True, slots=True)
class Workspace:
    """Obszar, w którym narzędzia plikowe mogą działać, wraz z limitami."""

    roots: tuple[Path, ...]
    max_read_bytes: int = 200_000
    max_write_bytes: int = 200_000
    max_entries: int = 200
    max_delete_entries: int = 50
    case_insensitive: bool = False

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, *, platform_info: PlatformInfo | None = None
    ) -> Workspace:
        active = settings or get_settings()
        info = platform_info or detect_platform()
        return cls(
            roots=tuple(_canonical(root) for root in configured_roots(active)),
            max_read_bytes=active.fs_max_read_bytes,
            max_write_bytes=active.fs_max_write_bytes,
            max_entries=active.fs_max_entries,
            max_delete_entries=active.fs_max_delete_entries,
            # Systemy plików Windowsa i macOS są (praktycznie zawsze) nieczułe na
            # wielkość liter; ext4 i pokrewne są czułe.
            case_insensitive=info.os_family in (OSFamily.WINDOWS, OSFamily.MACOS),
        )

    @classmethod
    def for_roots(cls, roots: Iterable[Path], **overrides: object) -> Workspace:
        """Obszar zbudowany wprost — używane w testach i przez narzędzia notatek."""
        values: dict[str, object] = {"roots": tuple(_canonical(root) for root in roots)}
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]

    # --- opis ------------------------------------------------------------- #

    @property
    def primary(self) -> Path:
        return self.roots[0]

    def describe(self) -> str:
        return ", ".join(str(root) for root in self.roots) or "brak"

    def exists(self) -> bool:
        return any(root.is_dir() for root in self.roots)

    def ensure_primary(self) -> Path | None:
        """Utwórz katalog roboczy, gdy go nie ma. ``None`` = nie udało się."""
        try:
            self.primary.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Nie udało się utworzyć katalogu %s: %s", self.primary, exc)
            return None
        return self.primary

    # --- sprawdzanie ścieżek ---------------------------------------------- #

    def contains(self, candidate: Path) -> bool:
        """Czy (już skanonizowana) ścieżka leży w dozwolonym obszarze?"""
        return any(_inside(candidate, root, self.case_insensitive) for root in self.roots)

    def is_root(self, candidate: Path) -> bool:
        """Czy ścieżka JEST jednym z dozwolonych katalogów (a nie czymś w nim)?"""
        return any(_same(candidate, root, self.case_insensitive) for root in self.roots)

    def resolve(
        self,
        raw: str,
        *,
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
    ) -> Path:
        """Zamień ścieżkę od modelu na bezpieczną, sprawdzoną ścieżkę bezwzględną.

        Rzuca :class:`PathNotAllowedError` z powodem nadającym się do pokazania — router
        zamieni go na zwykły wynik ``ok=false``.
        """
        text = str(raw or "").strip().strip('"').strip("'")
        if not text:
            raise PathNotAllowedError("nie podano ścieżki")

        expanded = Path(os.path.expandvars(text)).expanduser()
        if not expanded.is_absolute():
            # Ścieżka względna liczy się od katalogu roboczego ASYSTENTA, nie od
            # cwd procesu: cwd zależy od tego, skąd ktoś uruchomił program.
            # Wyjątek: gdy pierwszy segment to NAZWA dozwolonego katalogu, liczymy
            # od niego. Bez tego model, który odeśle ścieżkę zobaczoną w wyniku
            # („dokumenty/plan.txt"), trafiałby w <katalog>/dokumenty/plan.txt.
            expanded = self._base_for(expanded) / _strip_root_prefix(expanded, self.roots)

        candidate = _canonical(expanded)
        if not self.contains(candidate):
            raise PathNotAllowedError(
                f"ścieżka '{text}' jest poza dozwolonymi katalogami ({self.describe()}). "
                "Dostęp poza nie jest możliwy — także przez '..' i dowiązania symboliczne."
            )
        if must_exist and not candidate.exists():
            raise PathNotAllowedError(
                f"nie ma takiej ścieżki: {self.label(candidate)}. Ścieżki podawaj względem "
                "dozwolonego katalogu (np. 'plan.txt', 'notatki/rower.md'); '.' oznacza sam "
                "katalog. Listę katalogów da narzędzie fs.roots."
            )
        if must_be_file and candidate.exists() and not candidate.is_file():
            raise PathNotAllowedError(f"to nie jest plik: {self.label(candidate)}")
        if must_be_dir and candidate.exists() and not candidate.is_dir():
            raise PathNotAllowedError(f"to nie jest katalog: {self.label(candidate)}")
        return candidate

    def _base_for(self, relative: Path) -> Path:
        """Który dozwolony katalog jest punktem odniesienia dla tej ścieżki."""
        parts = relative.parts
        if parts:
            head = parts[0]
            for root in self.roots:
                if _same(Path(root.name), Path(head), self.case_insensitive):
                    return root
        return self.primary

    def label(self, path: Path) -> str:
        """Ścieżka do pokazania — w postaci, którą DA SIĘ odesłać z powrotem.

        To nie jest kosmetyka: model widzi etykietę w wyniku narzędzia i w
        następnym wywołaniu podaje ją jako ścieżkę. Dlatego przy jednym dozwolonym
        katalogu etykieta jest zwykłą ścieżką względną (``plan.txt``), a przy kilku
        dostaje z przodu nazwę katalogu (``dokumenty/plan.txt``) — i :meth:`resolve`
        rozumie oba zapisy. Pełnej ścieżki z nazwą użytkownika nie pokazujemy.
        """
        multiple = len(self.roots) > 1
        for root in self.roots:
            if _inside(path, root, self.case_insensitive):
                try:
                    relative = path.relative_to(root)
                except ValueError:  # pragma: no cover - różnica wielkości liter
                    continue
                inner = str(relative)
                if inner == ".":
                    return root.name
                return f"{root.name}/{inner}" if multiple else inner
        return str(path)


# --------------------------------------------------------------------------- #
# Kanonizacja i porównania
# --------------------------------------------------------------------------- #


def _strip_root_prefix(relative: Path, roots: Sequence[Path]) -> Path:
    """Zdejmij z początku nazwę dozwolonego katalogu, jeśli tam stoi.

    ``dokumenty/plan.txt`` → ``plan.txt``, gdy jednym z dozwolonych katalogów jest
    coś o nazwie ``dokumenty``. Sama nazwa katalogu (``dokumenty``) staje się ``.``.

    Poza nazwami katalogów przyjmujemy też słowo ``workspace``: modele wpisują je
    jako ścieżkę, bo tak nazywa się obszar roboczy w opisach narzędzi. Rozumienie
    tego zapisu kosztuje trzy linijki, a oszczędza całą turę rozmowy zmarnowaną na
    komunikat „nie ma takiej ścieżki".
    """
    parts = relative.parts
    if not parts:
        return relative
    head = parts[0]
    aliases = {root.name for root in roots} | {WORKSPACE_DIR_NAME}
    if head in aliases:
        rest = parts[1:]
        return Path(*rest) if rest else Path(".")
    return relative


def _canonical(path: Path) -> Path:
    """Ścieżka bezwzględna bez ``..`` i bez dowiązań symbolicznych.

    ``strict=False``: ścieżka nie musi istnieć (zapis nowego pliku), ale to, co
    z niej istnieje, jest rozwijane — więc dowiązanie w środku ścieżki nie
    przemyci nas poza dozwolony katalog.
    """
    try:
        return Path(os.path.realpath(str(path)))
    except OSError:  # pragma: no cover - zależne od systemu plików
        return path.absolute()


def _key(path: Path, case_insensitive: bool) -> str:
    text = str(path)
    return text.casefold() if case_insensitive else text


def _same(left: Path, right: Path, case_insensitive: bool) -> bool:
    return _key(left, case_insensitive) == _key(right, case_insensitive)


def _inside(candidate: Path, root: Path, case_insensitive: bool) -> bool:
    """Czy ``candidate`` to ``root`` albo coś w nim?

    Porównujemy po częściach ścieżki, a nie po prefiksie tekstu: ``/dane2`` nie
    jest w ``/dane``, choć tekstowo się nim zaczyna.
    """
    if _same(candidate, root, case_insensitive):
        return True
    candidate_parts = candidate.parts
    root_parts = root.parts
    if len(candidate_parts) <= len(root_parts):
        return False
    if case_insensitive:
        return all(
            left.casefold() == right.casefold()
            for left, right in zip(candidate_parts, root_parts, strict=False)
        )
    return candidate_parts[: len(root_parts)] == root_parts


# --------------------------------------------------------------------------- #
# Odczyt plików
# --------------------------------------------------------------------------- #


def looks_binary(path: Path) -> bool:
    """Czy plik wygląda na binarny (bajt zerowy w początkowym fragmencie)?

    Prosta heurystyka, ta sama, której używa ``git``. Wystarczy, żeby nie wsypać
    modelowi do promptu megabajta śmieci z pliku ``.onnx``.
    """
    try:
        with path.open("rb") as handle:
            return b"\x00" in handle.read(_BINARY_PROBE_BYTES)
    except OSError:  # pragma: no cover - zależne od uprawnień
        return False


def read_text_limited(path: Path, max_bytes: int) -> tuple[str, bool]:
    """Wczytaj tekst do ``max_bytes`` bajtów. Zwraca treść i znacznik obcięcia.

    Kodowanie: UTF-8 z zamianą niepoprawnych bajtów. **Nie** używamy kodowania
    domyślnego dla systemu — ten sam plik czytany na Windowsie (cp1250) i na
    Linuksie (UTF-8) dałby inną treść, a model dostawałby różne dane w zależności
    od maszyny.
    """
    try:
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError as exc:
        raise PathNotAllowedError(f"nie udało się odczytać pliku: {exc}") from exc

    truncated = len(raw) > max_bytes
    if truncated:
        raw = raw[:max_bytes]
    return raw.decode("utf-8", errors="replace"), truncated


def entry_info(path: Path, *, workspace: Workspace | None = None) -> dict[str, object]:
    """Opis wpisu katalogu: nazwa, rodzaj, rozmiar, data modyfikacji."""
    try:
        stat = path.stat()
        size = int(stat.st_size)
        modified = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(
            timespec="seconds"
        )
    except OSError:  # pragma: no cover - plik zniknął w trakcie listowania
        size, modified = 0, ""
    kind = "dir" if path.is_dir() else ("link" if path.is_symlink() else "file")
    return {
        "name": path.name,
        "kind": kind,
        "size": size,
        "modified": modified,
        "path": workspace.label(path) if workspace is not None else str(path),
    }


def count_tree(path: Path, *, limit: int) -> tuple[int, int]:
    """Ile plików i katalogów jest w drzewie (do ``limit`` wpisów).

    Używane przez potwierdzenie usunięcia: użytkownik ma zobaczyć, ile rzeczy
    zniknie, ZANIM się zgodzi.
    """
    files = 0
    directories = 0
    for current, dirnames, filenames in os.walk(path):
        directories += len(dirnames)
        files += len(filenames)
        if files + directories > limit:
            break
        del current  # tylko licznik — ścieżki nie są tu potrzebne
    return files, directories


def sorted_entries(directory: Path, *, limit: int) -> list[Path]:
    """Wpisy katalogu: najpierw katalogi, potem pliki, alfabetycznie.

    Sortowanie jest nasze, nie systemowe: kolejność z ``iterdir()`` zależy od
    systemu plików, a model i użytkownik mają widzieć zawsze to samo.
    """
    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        raise PathNotAllowedError(f"nie udało się odczytać katalogu: {exc}") from exc
    entries.sort(key=lambda item: (not item.is_dir(), item.name.casefold()))
    return entries[:limit]


def available_space(path: Path) -> int | None:
    """Wolne miejsce na dysku w bajtach albo ``None``, gdy nie da się sprawdzić."""
    import shutil

    try:
        return int(shutil.disk_usage(str(path)).free)
    except OSError:  # pragma: no cover - zależne od systemu plików
        return None


def workspace_from_settings(
    settings: Settings | None = None, *, extra_roots: Sequence[Path] = ()
) -> Workspace:
    """Obszar z konfiguracji, opcjonalnie rozszerzony o katalogi wskazane w kodzie."""
    workspace = Workspace.from_settings(settings)
    if not extra_roots:
        return workspace
    return Workspace(
        roots=workspace.roots + tuple(_canonical(root) for root in extra_roots),
        max_read_bytes=workspace.max_read_bytes,
        max_write_bytes=workspace.max_write_bytes,
        max_entries=workspace.max_entries,
        max_delete_entries=workspace.max_delete_entries,
        case_insensitive=workspace.case_insensitive,
    )


def notes_directory(settings: Settings | None = None) -> Path:
    """Katalog na notatki zapisywane jako pliki (``workspace/notes``)."""
    return default_workspace(settings) / "notes"


def env_override_roots() -> list[Path]:
    """Katalogi wskazane zmiennymi środowiskowymi instalatora (jeśli są)."""
    roots: list[Path] = []
    for variable in ("MIKU_FS_ROOT", "MIKU_WORKSPACE_DIR"):
        candidate = path_from_env(variable)
        if candidate is not None:
            roots.append(candidate)
    return roots


__all__ = [
    "WORKSPACE_DIR_NAME",
    "PathNotAllowedError",
    "Workspace",
    "available_space",
    "configured_roots",
    "count_tree",
    "default_workspace",
    "entry_info",
    "looks_binary",
    "notes_directory",
    "read_text_limited",
    "sorted_entries",
    "workspace_from_settings",
]
