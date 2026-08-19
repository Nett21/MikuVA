"""Ograniczenie historii przekazywanej do modelu.

Okno rozmowy (``ConversationHistory``) i to, co realnie leci do modelu, to dwie
różne wielkości. Okno jest większe, bo z niego powstają streszczenia i to ono
opisuje rozmowę dla człowieka; do modelu idzie ostatni fragment, bo na słabszej
maszynie każdy tysiąc tokenów promptu to sekundy czekania.

Testy pilnują trzech rzeczy, z których każda była realnym błędem w tej klasie
rozwiązań:

* bieżące pytanie NIGDY nie wypada — inaczej model odpowiada na poprzednie,
* wynik narzędzia nie zostaje bez swojego wywołania — inaczej model dostaje
  wynik „znikąd" i albo powtarza wywołanie, albo opowiada o akcji, która się nie
  wykonała,
* limit dotyczy ładunku HTTP, a nie tylko funkcji pomocniczej — inaczej łatwo
  o wersję, w której helper działa, ale nikt go nie woła.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from brain.conversation import ConversationHistory, Message, select_for_model
from brain.llm import OllamaClient
from config import Settings


def make(role: str, content: str, **metadata: Any) -> Message:
    return Message(role=role, content=content, metadata=metadata)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# select_for_model
# --------------------------------------------------------------------------- #


def test_bez_limitow_przechodzi_wszystko() -> None:
    messages = [make("user", "a"), make("assistant", "b"), make("user", "c")]
    assert select_for_model(messages) == tuple(messages)
    assert select_for_model(messages, max_messages=0, max_chars=0) == tuple(messages)


def test_pusta_historia_nie_wywraca_sie() -> None:
    assert select_for_model([]) == ()
    assert select_for_model([], max_messages=5, max_chars=100) == ()


def test_limit_liczby_wiadomosci_zostawia_ogon() -> None:
    messages = [make("user", str(index)) for index in range(10)]
    wybrane = select_for_model(messages, max_messages=3)
    assert [message.content for message in wybrane] == ["7", "8", "9"]


def test_limit_znakow_zostawia_ogon() -> None:
    messages = [make("user", "x" * 100) for _ in range(10)]
    wybrane = select_for_model(messages, max_chars=250)
    assert len(wybrane) == 2  # 2 × 100 mieści się w 250, trzecia już nie


def test_biezaca_tura_zostaje_nawet_gdy_sama_przekracza_limit() -> None:
    """Pytanie dłuższe niż cały limit i tak musi dojść do modelu.

    Alternatywa — pusta lista wiadomości — kończy się odpowiedzią na nic.
    """
    messages = [make("user", "stare"), make("user", "y" * 5_000)]
    wybrane = select_for_model(messages, max_chars=100)
    assert len(wybrane) == 1
    assert wybrane[0].content == "y" * 5_000


def test_osierocony_wynik_narzedzia_nie_trafia_do_modelu() -> None:
    """Wiadomość ``tool`` bez poprzedzającego ją wywołania jest odcinana."""
    messages = [
        make("user", "usuń plik"),
        make("assistant", "", tool_calls=[{"function": {"name": "fs.delete"}}]),
        make("tool", "usunięto", tool="fs.delete"),
        make("assistant", "gotowe"),
        make("user", "dzięki"),
    ]
    # Limit 3 trafiłby dokładnie w „tool" jako pierwszą wiadomość.
    wybrane = select_for_model(messages, max_messages=3)
    assert [message.role for message in wybrane] == ["assistant", "user"]
    assert wybrane[0].content == "gotowe"


def test_para_wywolanie_wynik_przechodzi_w_calosci_gdy_sie_miesci() -> None:
    messages = [
        make("user", "usuń plik"),
        make("assistant", "", tool_calls=[{"function": {"name": "fs.delete"}}]),
        make("tool", "usunięto", tool="fs.delete"),
        make("assistant", "gotowe"),
    ]
    wybrane = select_for_model(messages, max_messages=4)
    assert [message.role for message in wybrane] == ["user", "assistant", "tool", "assistant"]


def test_same_wyniki_narzedzi_nie_kasuja_ostatniej_wiadomosci() -> None:
    """Przypadek skrajny: ogon złożony wyłącznie z wiadomości ``tool``.

    Odcinanie osieroconych wyników nie może zjeść wszystkiego — model musi coś
    dostać, inaczej Ollama odrzuca żądanie z pustą listą wiadomości.
    """
    messages = [make("tool", "a", tool="x"), make("tool", "b", tool="x")]
    wybrane = select_for_model(messages, max_messages=2)
    assert len(wybrane) == 1
    assert wybrane[0].content == "b"


def test_kolejnosc_jest_zachowana() -> None:
    messages = [make("user", str(index)) for index in range(6)]
    wybrane = select_for_model(messages, max_messages=4)
    assert [message.content for message in wybrane] == ["2", "3", "4", "5"]


# --------------------------------------------------------------------------- #
# Ładunek wysyłany do Ollamy
# --------------------------------------------------------------------------- #


@pytest.fixture
def klient(tmp_path: Any) -> OllamaClient:
    settings = Settings(
        _env_file=None,
        llm_history_max_messages=4,
        llm_history_max_chars=0,
        piper_voices_dir=str(tmp_path / "voices"),
    )
    return OllamaClient(settings)


def test_payload_stosuje_limit_historii(klient: OllamaClient) -> None:
    history = ConversationHistory(max_messages=100, max_chars=100_000)
    for index in range(12):
        history.add_user(f"pytanie {index}")

    payload = klient._payload(
        history.messages, stream=False, system="prompt"
    )
    role = [message["role"] for message in payload["messages"]]
    assert role[0] == "system"
    # 1 system + 4 wiadomości okna (limit), reszta zostaje w oknie, ale nie leci
    assert len(payload["messages"]) == 5
    assert payload["messages"][-1]["content"] == "pytanie 11"


def test_payload_bez_limitu_wysyla_cale_okno(tmp_path: Any) -> None:
    settings = Settings(
        _env_file=None,
        llm_history_max_messages=0,
        llm_history_max_chars=0,
        piper_voices_dir=str(tmp_path / "voices"),
    )
    client = OllamaClient(settings)
    history = ConversationHistory(max_messages=100, max_chars=100_000)
    for index in range(12):
        history.add_user(f"pytanie {index}")

    payload = client._payload(history.messages, stream=False, system=None)
    assert len(payload["messages"]) == 12


def test_limit_nie_dotyczy_bloku_kontekstu(klient: OllamaClient) -> None:
    """Kontekst tury (fakty, wspomnienia, godzina) idzie ZAWSZE.

    To on niesie streszczenie starszych tur — gdyby wypadał razem z historią,
    ograniczenie historii kasowałoby pamięć zamiast ją zastępować.
    """
    history = ConversationHistory(max_messages=100, max_chars=100_000)
    for index in range(12):
        history.add_user(f"pytanie {index}")

    payload = klient._payload(
        history.messages, stream=False, system="prompt", context="=== pamięć ==="
    )
    assert payload["messages"][-1]["content"] == "=== pamięć ==="
    assert payload["messages"][-2]["content"] == "pytanie 11"


def test_payload_jest_serializowalny_do_json(klient: OllamaClient) -> None:
    """Cały ładunek musi przejść przez ``json.dumps`` — httpx zrobi to samo."""
    history = ConversationHistory(max_messages=100, max_chars=100_000)
    history.add_user("cześć")
    history.add_assistant("", tool_calls=[{"function": {"name": "time.now", "arguments": {}}}])
    history.add_tool("12:00", tool="time.now")
    history.add_user("dzięki")

    payload = klient._payload(history.messages, stream=True, system="prompt")
    assert json.loads(json.dumps(payload))["stream"] is True


def test_domyslne_ustawienia_maja_limit_mniejszy_niz_okno() -> None:
    """Domyślnie do modelu idzie MNIEJ niż mieści okno — inaczej opcja nic nie daje."""
    settings = Settings(_env_file=None)
    assert 0 < settings.llm_history_max_messages < settings.history_max_messages
    assert 0 < settings.llm_history_max_chars < settings.history_max_chars
