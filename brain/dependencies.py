"""Sprawdzenia pamięci semantycznej dopięte do mechanizmu z Fazy 1.

Import tego modułu dokłada pozycje do ``python main.py --check-deps`` i do
``config/dependency_status.json``. Żadna z nich nie jest wymagana: brak modelu
embeddingów ma wyłączyć wyszukiwanie po znaczeniu, a nie zatrzymać asystenta.

Sprawdzenia są **tanie i bez skutków ubocznych**: nie ładują modelu (to kilka
sekund i setki megabajtów), nie pobierają niczego i nie odpytują Ollamy —
diagnostyka ma opisywać maszynę, a nie zmieniać jej stan.
"""

from __future__ import annotations

import importlib.util
import logging

from config import (
    EMBEDDINGS_DIR,
    DependencyCheck,
    DependencyContext,
    find_local_embedding_model,
    pip_install_hint,
    register_dependency_check,
)
from i18n import t

logger = logging.getLogger(__name__)


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _check_engine(context: DependencyContext) -> DependencyCheck:
    """Czym ta maszyna policzy embeddingi (i czy w ogóle)."""
    settings = context.settings
    if not settings.embeddings_enabled or settings.embedding_engine == "none":
        return DependencyCheck(
            name=t("deps.sem.name"),
            category="model",
            required=False,
            ok=False,
            detail=t("deps.sem.disabled"),
            hint=t("deps.sem.disabled_hint"),
            phase=6,
        )

    has_st = _module_available("sentence_transformers")
    wants_st = settings.embedding_engine in ("auto", "sentence-transformers")

    if wants_st and has_st:
        local = find_local_embedding_model(settings.embedding_model, EMBEDDINGS_DIR)
        return DependencyCheck(
            name=t("deps.sem.name"),
            category="model",
            required=False,
            ok=True,
            detail=(
                t("deps.sem.st_model", model=settings.embedding_model)
                + (t("deps.sem.on_disk") if local else t("deps.sem.will_download"))
            ),
            path=str(local) if local else str(EMBEDDINGS_DIR),
            hint=(
                ""
                if local
                else t("deps.sem.download_hint")
            ),
            phase=6,
        )

    if settings.embedding_engine == "sentence-transformers":
        return DependencyCheck(
            name=t("deps.sem.name"),
            category="model",
            required=False,
            ok=False,
            detail=t("deps.sem.st_missing"),
            hint=t("deps.sem.st_missing_hint", install=pip_install_hint(context.offline)),
            phase=6,
        )

    # Wariant z Ollamą: usługi tutaj NIE odpytujemy (to żądanie HTTP i osobna
    # pozycja w raporcie). Sprawdzamy tylko, czy model jest wśród pobranych.
    model = settings.embedding_ollama_model
    if not context.ollama.reachable:
        detail = t("deps.sem.ollama_down")
        ok = False
    else:
        ok = any(name.split(":")[0] == model.split(":")[0] for name in context.ollama.models)
        detail = (
            t("deps.sem.ollama_ok", model=model)
            if ok
            else t("deps.sem.ollama_missing_model", model=model)
        )
    return DependencyCheck(
        name=t("deps.sem.name"),
        category="model",
        required=False,
        ok=ok,
        detail=detail,
        path=context.settings.ollama_host,
        hint="" if ok else t("deps.sem.ollama_hint", model=model),
        phase=6,
    )


def _check_vector_backend(context: DependencyContext) -> DependencyCheck:
    """Czym liczone jest podobieństwo wektorów."""
    has_numpy = _module_available("numpy")
    has_faiss = _module_available("faiss")

    if not has_numpy:
        return DependencyCheck(
            name=t("deps.vector.name"),
            category="package",
            required=False,
            ok=False,
            detail=t("deps.vector.no_numpy"),
            hint=pip_install_hint(context.offline),
            phase=6,
        )
    return DependencyCheck(
        name=t("deps.vector.name"),
        category="package",
        required=False,
        ok=True,
        detail=t("deps.vector.faiss") if has_faiss else t("deps.vector.numpy"),
        phase=6,
    )


def _check_index_state(context: DependencyContext) -> DependencyCheck | None:
    """Ile wektorów leży w bazie. Bez otwierania bazy, gdy pamięć jest wyłączona."""
    if not context.settings.memory_enabled or not context.settings.embeddings_enabled:
        return None

    from config import MEMORY_DATABASE, database_file

    target = database_file(context.settings)
    if str(target) != MEMORY_DATABASE and not target.exists():
        return DependencyCheck(
            name=t("deps.index.name"),
            category="storage",
            required=False,
            ok=True,
            detail=t("deps.index.no_database"),
            phase=6,
        )

    try:
        import sqlite3

        connection = sqlite3.connect(str(target))
        try:
            row = connection.execute(
                "SELECT model, COUNT(*) FROM embeddings GROUP BY model ORDER BY COUNT(*) DESC"
            ).fetchone()
        finally:
            connection.close()
    except Exception as exc:  # tabeli może nie być (starsza baza) — to nie błąd
        logger.debug("Nie odczytano stanu indeksu wspomnień: %s", exc)
        return DependencyCheck(
            name=t("deps.index.name"),
            category="storage",
            required=False,
            ok=True,
            detail=t("deps.index.empty_db"),
            path=str(target),
            phase=6,
        )

    if row is None:
        detail = t("deps.index.empty")
    else:
        detail = t("deps.index.count", count=int(row[1]), model=row[0])
    return DependencyCheck(
        name=t("deps.index.name"),
        category="storage",
        required=False,
        ok=True,
        detail=detail,
        path=str(target),
        phase=6,
    )


@register_dependency_check
def check_semantic_memory(context: DependencyContext) -> list[DependencyCheck]:
    """Sprawdzenia Fazy 6 zebrane w jedną pozycję rejestru."""
    checks: list[DependencyCheck] = [_check_engine(context), _check_vector_backend(context)]
    state = _check_index_state(context)
    if state is not None:
        checks.append(state)
    return checks


__all__ = ["check_semantic_memory"]
