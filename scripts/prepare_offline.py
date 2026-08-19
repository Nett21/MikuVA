#!/usr/bin/env python3
"""Przygotowanie projektu do pracy BEZ internetu.

Ten skrypt jest jedynym miejscem, które celowo korzysta z sieci. Uruchamiasz go
raz, na maszynie z internetem — potem cały asystent działa offline.

Pobiera pięć rzeczy, których nie da się wytworzyć lokalnie:

1. **koła pip** do ``vendor/wheels`` — instalacja zależności na maszynie bez sieci,
2. **model Whisper** do ``models/whisper`` — rozpoznawanie mowy,
3. **głos Pipera** do ``models/piper`` — mówienie odpowiedzi,
4. **model embeddingów** do ``models/embeddings`` — pamięć semantyczna (Faza 6),
5. **model językowy Ollamy** (``ollama pull``) — sama rozmowa.

Użycie::

    python scripts/prepare_offline.py                 # wszystko naraz
    python scripts/prepare_offline.py --whisper       # tylko model mowy
    python scripts/prepare_offline.py --piper-voice pl_PL-darkman-medium
    python scripts/prepare_offline.py --embeddings    # tylko model pamięci (Faza 6)
    python scripts/prepare_offline.py --wheels --dev  # koła razem z pakietami testowymi
    python scripts/prepare_offline.py --whisper-model small --whisper-model tiny
    python scripts/prepare_offline.py --check         # sam audyt, bez sieci

``--check`` nie pobiera niczego: wypisuje, czego brakuje do pracy offline, i
zwraca kod wyjścia ``1``, jeśli komplet nie jest gotowy — nadaje się do skryptów
instalacyjnych i do CI.

Uwaga o kołach pip: są budowane dla konkretnego systemu i wersji Pythona.
Magazyn przygotowany na Linuksie z Pythonem 3.12 nie zadziała na Windowsie ani
na Pythonie 3.14 — w takim wypadku użyj opcji ``--pip-platform``/``--pip-python``
albo przygotuj magazyn na maszynie docelowej.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess  # nosec B404 - wołamy wyłącznie pip i ollama, bez powłoki
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402 - po ustawieniu sys.path
    APP_VERSION,
    EMBEDDINGS_DIR,
    OFFLINE_BUNDLE_FILE,
    PIPER_DIR,
    TAG_ERROR,
    TAG_SYSTEM,
    WHEELHOUSE_DIR,
    WHISPER_CACHE_DIR,
    detect_ollama,
    ensure_directories,
    find_local_embedding_model,
    find_local_whisper_model,
    get_settings,
    get_user_settings,
    iter_local_whisper_models,
    wheelhouse_packages,
)

REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
REQUIREMENTS_DEV = PROJECT_ROOT / "requirements-dev.txt"

EXIT_OK = 0
EXIT_INCOMPLETE = 1


# --------------------------------------------------------------------------- #
# Kroki
# --------------------------------------------------------------------------- #


def _run(command: Sequence[str], *, description: str) -> bool:
    """Uruchom polecenie, pokazując jego wyjście na bieżąco."""
    print(f"{TAG_SYSTEM} {description}")
    print(f"        $ {' '.join(command)}")
    try:
        completed = subprocess.run(command, check=False)  # nosec B603 - lista argumentów, bez powłoki
    except OSError as exc:
        print(f"{TAG_ERROR} Nie udało się uruchomić polecenia: {exc}")
        return False
    if completed.returncode != 0:
        print(f"{TAG_ERROR} Polecenie zakończyło się kodem {completed.returncode}.")
        return False
    return True


def download_wheels(
    *, dev: bool, pip_platform: str | None, pip_python: str | None
) -> bool:
    """Zbierz koła pip do ``vendor/wheels``."""
    WHEELHOUSE_DIR.mkdir(parents=True, exist_ok=True)

    requirement_files = [REQUIREMENTS] + ([REQUIREMENTS_DEV] if dev else [])
    for path in requirement_files:
        if not path.is_file():
            print(f"{TAG_ERROR} Brak pliku {path}.")
            return False

    command = [sys.executable, "-m", "pip", "download", "--dest", str(WHEELHOUSE_DIR)]
    if pip_platform or pip_python:
        # Pobieranie dla innego systemu wymaga gotowych kół — pip nie zbuduje
        # źródeł pod obcą platformę.
        command += ["--only-binary=:all:"]
        if pip_platform:
            command += ["--platform", pip_platform]
        if pip_python:
            command += ["--python-version", pip_python]
    for path in requirement_files:
        command += ["-r", str(path)]

    if not _run(command, description=f"Pobieram pakiety pip do {WHEELHOUSE_DIR}"):
        return False

    # webrtcvad jest w requirements.txt tylko jako komentarz (nie ma kół dla
    # każdej wersji Pythona), ale jeśli akurat są — warto je mieć offline.
    optional = [
        sys.executable, "-m", "pip", "download",
        "--dest", str(WHEELHOUSE_DIR), "--only-binary=:all:", "webrtcvad-wheels",
    ]
    if not _run(optional, description="Pobieram opcjonalny pakiet webrtcvad-wheels"):
        print(f"{TAG_SYSTEM} Pomijam webrtcvad — bez niego działa wbudowany VAD energetyczny.")

    print(f"{TAG_SYSTEM} W magazynie jest {wheelhouse_packages()} pakietów.")
    return True


def download_whisper(models: Sequence[str]) -> bool:
    """Pobierz modele Whisper do ``models/whisper``."""
    try:
        from faster_whisper import download_model  # noqa: PLC0415 - import leniwy
    except ImportError:
        print(
            f"{TAG_ERROR} Pakiet faster-whisper nie jest zainstalowany — "
            "najpierw zainstaluj zależności (python -m pip install -r requirements.txt)."
        )
        return False

    WHISPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for model in models:
        if find_local_whisper_model(model, WHISPER_CACHE_DIR) is not None:
            print(f"{TAG_SYSTEM} Model Whisper '{model}' już jest na dysku — pomijam.")
            continue
        print(f"{TAG_SYSTEM} Pobieram model Whisper '{model}' do {WHISPER_CACHE_DIR}...")
        try:
            path = download_model(model, cache_dir=str(WHISPER_CACHE_DIR))
        except Exception as exc:  # sieć, brak repozytorium, błędna nazwa
            print(f"{TAG_ERROR} Nie udało się pobrać modelu '{model}': {exc}")
            ok = False
            continue
        print(f"{TAG_SYSTEM} Gotowe: {path}")
    return ok


def download_piper_voices(voices: Sequence[str]) -> bool:
    """Pobierz głosy Pipera do katalogu ``models/piper``.

    Pobieraniem zajmuje się narzędzie z pakietu ``piper-tts``
    (``python -m piper.download_voices``) — nie zaszywamy tu żadnego adresu URL
    ani układu katalogów repozytorium z głosami. Gdy pakietu nie ma, wypisujemy,
    co zrobić ręcznie: głos to dwa pliki, które wystarczy wrzucić do katalogu.
    """
    if not voices:
        print(
            f"{TAG_SYSTEM} Nie podano głosu do pobrania. Wskaż go opcją --piper-voice, np.:\n"
            "        python scripts/prepare_offline.py --piper-voice pl_PL-darkman-medium"
        )
        return True

    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for voice in voices:
        if (PIPER_DIR / f"{voice}.onnx").is_file():
            print(f"{TAG_SYSTEM} Głos '{voice}' już jest na dysku — pomijam.")
            continue
        downloaded = _run(
            [
                sys.executable, "-m", "piper.download_voices", voice,
                "--download-dir", str(PIPER_DIR),
            ],
            description=f"Pobieram głos '{voice}' do {PIPER_DIR}",
        )
        if not downloaded:
            ok = False
            print(
                f"{TAG_ERROR} Nie udało się pobrać głosu '{voice}'.\n"
                "        Zainstaluj pakiet (pip install piper-tts) albo pobierz ręcznie dwa\n"
                f"        pliki — {voice}.onnx i {voice}.onnx.json — z repozytorium głosów\n"
                f"        Pipera (rhasspy/piper-voices) i wrzuć je do {PIPER_DIR}."
            )
    return ok


def download_embedding_model(model: str) -> bool:
    """Pobierz model embeddingów do ``models/embeddings`` (Faza 6).

    Pobieraniem zajmuje się sama biblioteka ``sentence-transformers`` — nie ma tu
    zaszytego adresu URL ani układu katalogów repozytorium. Wystarczy raz
    utworzyć obiekt modelu: biblioteka ściąga wagi do wskazanego katalogu.
    """
    if find_local_embedding_model(model, EMBEDDINGS_DIR) is not None:
        print(f"{TAG_SYSTEM} Model embeddingów '{model}' już jest na dysku — pomijam.")
        return True

    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - import leniwy
    except ImportError:
        print(
            f"{TAG_ERROR} Pakiet sentence-transformers nie jest zainstalowany.\n"
            "        Zainstaluj go (pip install sentence-transformers) albo korzystaj z\n"
            "        embeddingów Ollamy: EMBEDDING_ENGINE=ollama w .env, a potem\n"
            "        ollama pull nomic-embed-text"
        )
        return False

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{TAG_SYSTEM} Pobieram model embeddingów '{model}' do {EMBEDDINGS_DIR}...")
    try:
        SentenceTransformer(model, cache_folder=str(EMBEDDINGS_DIR))
    except Exception as exc:  # sieć, błędna nazwa, brak miejsca
        print(f"{TAG_ERROR} Nie udało się pobrać modelu '{model}': {exc}")
        return False

    # Sprawdzamy, czy pliki naprawdę leżą tam, gdzie ich potem szukamy. Nie jest
    # to nadmierna ostrożność: o miejsce zapisu spierają się zmienne środowiskowe
    # (HF_HOME, HF_HUB_CACHE, SENTENCE_TRANSFORMERS_HOME) i wersja biblioteki, a
    # bez tej kontroli maszyna offline dowiedziałaby się o problemie dopiero
    # wtedy, gdy nie ma już skąd pobrać modelu.
    located = find_local_embedding_model(model, EMBEDDINGS_DIR)
    if located is None:
        print(
            f"{TAG_ERROR} Pobranie się udało, ale modelu nie ma w {EMBEDDINGS_DIR}.\n"
            "        Biblioteka zapisała go gdzie indziej (decydują zmienne HF_HOME,\n"
            "        HF_HUB_CACHE i SENTENCE_TRANSFORMERS_HOME). Przenieś ten katalog\n"
            f"        do {EMBEDDINGS_DIR} albo wskaż go wprost w .env:\n"
            "        EMBEDDING_MODEL=/pelna/sciezka/do/katalogu/modelu"
        )
        return False
    print(f"{TAG_SYSTEM} Gotowe: {located}")
    return True


def pull_ollama_model(model: str) -> bool:
    """Pobierz model językowy przez ``ollama pull``."""
    binary = shutil.which("ollama")
    if binary is None:
        print(
            f"{TAG_ERROR} Nie znaleziono polecenia 'ollama' w PATH — "
            f"zainstaluj Ollamę i wykonaj: ollama pull {model}"
        )
        return False
    return _run([binary, "pull", model], description=f"Pobieram model językowy '{model}'")


# --------------------------------------------------------------------------- #
# Audyt gotowości
# --------------------------------------------------------------------------- #


def _local_piper_voices() -> list[str]:
    """Nazwy głosów Pipera leżących na dysku (bez importowania warstwy audio)."""
    try:
        return sorted(path.name.removesuffix(".onnx") for path in PIPER_DIR.glob("**/*.onnx"))
    except OSError:  # pragma: no cover - zależne od uprawnień
        return []


def audit(models: Sequence[str]) -> tuple[bool, dict[str, object]]:
    """Sprawdź (bez sieci), czy komplet do pracy offline jest na miejscu."""
    settings = get_settings()
    ollama = detect_ollama(settings)

    whisper_present = {
        model: find_local_whisper_model(model, WHISPER_CACHE_DIR) for model in models
    }
    wheels = wheelhouse_packages()
    piper_voices = _local_piper_voices()
    embedding_model = (
        find_local_embedding_model(settings.embedding_model, EMBEDDINGS_DIR)
        if settings.embeddings_enabled
        else None
    )

    print()
    print("Gotowość do pracy bez internetu:")
    for model, path in whisper_present.items():
        status = "ok  " if path is not None else "BRAK"
        print(f"  Model Whisper '{model}'".ljust(34) + f"{status}  {path or WHISPER_CACHE_DIR}")
    print(
        "  Głos Piper (mowa)".ljust(34)
        + (
            f"ok    {', '.join(piper_voices)}"
            if piper_voices
            else f"BRAK  brak plików .onnx w {PIPER_DIR}"
        )
    )
    if settings.embeddings_enabled:
        print(
            "  Model embeddingów (pamięć)".ljust(34)
            + (
                f"ok    {embedding_model}"
                if embedding_model is not None
                else f"BRAK  {settings.embedding_model} — nie ma go w {EMBEDDINGS_DIR}"
            )
        )
    print(
        "  Koła pip (vendor/wheels)".ljust(34)
        + f"{'ok  ' if wheels else 'BRAK'}  {wheels} pakietów w {WHEELHOUSE_DIR}"
    )
    print(
        f"  Model LLM '{settings.ollama_model}'".ljust(34)
        + (
            f"{'ok  ' if ollama.model_present else 'BRAK'}  {ollama.host}"
            if ollama.reachable
            else f"?     Ollama nie odpowiada ({ollama.error or 'brak odpowiedzi'})"
        )
    )

    other = sorted(iter_local_whisper_models(WHISPER_CACHE_DIR))
    if other:
        print(f"  Modele Whisper na dysku: {', '.join(other)}")
    print()

    # Magazyn kół nie wchodzi do oceny gotowości: przydaje się dopiero przy
    # instalacji na innej maszynie, a nie do uruchomienia na tej.
    ready = all(path is not None for path in whisper_present.values()) and (
        ollama.model_present or not ollama.reachable
    )
    if not wheels:
        print(
            f"{TAG_SYSTEM} Magazyn kół jest pusty — potrzebny tylko wtedy, gdy zależności "
            "trzeba będzie zainstalować na maszynie bez sieci."
        )
    if not ollama.reachable:
        print(
            f"{TAG_SYSTEM} Nie sprawdzono modelu językowego: Ollama nie odpowiada. "
            "Uruchom `ollama serve` i powtórz, żeby mieć pewność."
        )
    if settings.embeddings_enabled and embedding_model is None:
        print(
            f"{TAG_SYSTEM} Bez modelu embeddingów asystent działa, tylko nie kojarzy "
            "wspomnień po znaczeniu. Pobierz go: "
            "python scripts/prepare_offline.py --embeddings"
        )
    if not piper_voices:
        print(
            f"{TAG_SYSTEM} Bez głosu Pipera asystent działa, tylko nie mówi. "
            "Pobierz go: python scripts/prepare_offline.py --piper-voice pl_PL-darkman-medium"
        )

    manifest: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "app_version": APP_VERSION,
        "python": sys.version.split()[0],
        "ready": ready,
        "whisper_models": {
            model: str(path) if path else None for model, path in whisper_present.items()
        },
        "piper_voices": {"path": str(PIPER_DIR), "voices": piper_voices},
        "embedding_model": {
            "path": str(EMBEDDINGS_DIR),
            "name": settings.embedding_model,
            "present": embedding_model is not None,
        },
        "wheelhouse": {"path": str(WHEELHOUSE_DIR), "packages": wheels},
        "ollama": {
            "host": ollama.host,
            "reachable": ollama.reachable,
            "model": settings.ollama_model,
            "model_present": ollama.model_present,
        },
    }
    return ready, manifest


def write_manifest(manifest: dict[str, object]) -> None:
    try:
        OFFLINE_BUNDLE_FILE.parent.mkdir(parents=True, exist_ok=True)
        OFFLINE_BUNDLE_FILE.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"{TAG_ERROR} Nie udało się zapisać {OFFLINE_BUNDLE_FILE}: {exc}")
        return
    print(f"{TAG_SYSTEM} Manifest zapisany: {OFFLINE_BUNDLE_FILE}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _default_models(settings: Any) -> list[str]:
    """Modele Whispera potrzebne do pracy offline: główny + detektor frazy.

    Detektor słowa aktywującego (Faza 3) używa własnego, małego modelu. Bez
    niego bramka schodzi na model główny — działa, ale wolniej, więc pobieramy
    oba za jednym razem.
    """
    models = [settings.whisper_model]
    wake_model = settings.wake_whisper_model.strip()
    if settings.wake_enabled and wake_model and wake_model not in models:
        models.append(wake_model)
    return models


def _default_piper_voices() -> list[str]:
    """Głos wskazany przez użytkownika w ``config/user_settings.json``.

    Świadomie NIE podstawiamy tu żadnej nazwy "na wszelki wypadek": pobieranie
    kilkudziesięciu megabajtów cudzego wyboru bez pytania byłoby niegrzeczne.
    Gdy pole jest puste, skrypt mówi, jak wskazać głos.
    """
    model = get_user_settings().piper_model.strip()
    if not model or model.lower().endswith(".onnx") or "/" in model or "\\" in model:
        return []
    return [model]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_offline.py",
        description="Pobierz wszystko, czego asystent potrzebuje do pracy bez internetu.",
    )
    parser.add_argument("--wheels", action="store_true", help="pobierz koła pip do vendor/wheels")
    parser.add_argument("--whisper", action="store_true", help="pobierz model rozpoznawania mowy")
    parser.add_argument(
        "--ollama", action="store_true", help="pobierz model językowy (ollama pull)"
    )
    parser.add_argument(
        "--piper", action="store_true", help="pobierz głos do syntezy mowy (Faza 4)"
    )
    parser.add_argument(
        "--piper-voice",
        action="append",
        default=None,
        metavar="NAZWA",
        help=(
            "głos Pipera do pobrania, np. pl_PL-darkman-medium albo en_US-amy-medium; "
            "można podać wielokrotnie (domyślnie: głos z config/user_settings.json)"
        ),
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="pobierz model embeddingów do pamięci semantycznej (Faza 6)",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        metavar="NAZWA",
        help="model embeddingów do pobrania (domyślnie: EMBEDDING_MODEL z .env)",
    )
    parser.add_argument(
        "--whisper-model",
        action="append",
        default=None,
        metavar="NAZWA",
        help=(
            "model Whisper do pobrania (domyślnie: główny + model detektora frazy); "
            "można podać wielokrotnie"
        ),
    )
    parser.add_argument(
        "--dev", action="store_true", help="dołóż pakiety z requirements-dev.txt (testy)"
    )
    parser.add_argument(
        "--pip-platform",
        default=None,
        metavar="TAG",
        help="pobierz koła dla innej platformy, np. win_amd64 albo manylinux2014_x86_64",
    )
    parser.add_argument(
        "--pip-python",
        default=None,
        metavar="WERSJA",
        help="pobierz koła dla innej wersji Pythona, np. 3.12",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="tylko sprawdź gotowość do pracy offline, niczego nie pobieraj",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_directories()

    settings = get_settings()
    models = args.whisper_model or _default_models(settings)

    if args.check:
        ready, manifest = audit(models)
        write_manifest(manifest)
        if ready:
            print(f"{TAG_SYSTEM} Modele są na miejscu — asystent uruchomi się bez internetu.")
            print(f"{TAG_SYSTEM} Wymuś tryb offline: python main.py --offline")
            return EXIT_OK
        print(f"{TAG_ERROR} Czegoś brakuje — uruchom: python scripts/prepare_offline.py")
        return EXIT_INCOMPLETE

    # Brak wyboru = zrób wszystko.
    everything = not (
        args.wheels or args.whisper or args.ollama or args.piper or args.embeddings
    )
    ok = True

    if everything or args.wheels:
        ok = download_wheels(
            dev=args.dev, pip_platform=args.pip_platform, pip_python=args.pip_python
        ) and ok
    if everything or args.whisper:
        ok = download_whisper(models) and ok
    if everything or args.piper:
        ok = download_piper_voices(args.piper_voice or _default_piper_voices()) and ok
    if everything or args.embeddings:
        ok = download_embedding_model(args.embedding_model or settings.embedding_model) and ok
    if everything or args.ollama:
        ok = pull_ollama_model(settings.ollama_model) and ok

    ready, manifest = audit(models)
    write_manifest(manifest)

    if ok and ready:
        print(f"{TAG_SYSTEM} Gotowe. Od teraz: python main.py --offline")
        return EXIT_OK
    print(f"{TAG_ERROR} Przygotowanie nie jest kompletne — szczegóły powyżej.")
    return EXIT_INCOMPLETE


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:  # pragma: no cover - przerwanie przez użytkownika
        print()
        raise SystemExit(EXIT_INCOMPLETE) from None
