"""Testy tego, co decyduje o czasie oczekiwania na odpowiedź.

Serwer modelu (llama.cpp pod Ollamą) trzyma policzony **prefiks** promptu i przy
kolejnej turze przelicza dopiero od pierwszej różnicy. Deklaracje narzędzi to
~3400 tokenów i są renderowane razem z blokiem systemowym, więc **każda zmiana w
prompcie systemowym unieważnia je wszystkie**.

Zmierzone na maszynie użytkownika (qwen2.5:7b-instruct, Ollama licząca na CPU),
trzy tury pod rząd:

===============================================  =======================
układ promptu                                    czas czytania promptu
===============================================  =======================
zmienny prompt systemowy (znacznik czasu)        41–48 s w KAŻDEJ turze
kontekst jako wiadomość ``system`` na końcu      40–44 s w KAŻDEJ turze
kontekst jako wiadomość ``user`` na końcu        39 s, potem 3–6 s
===============================================  =======================

Ta różnica nie jest widoczna w kodzie „na oko" i łatwo ją cofnąć jedną
poprawką w prompcie. Stąd te testy: pilnują trzech własności, z których każda
osobno psuje czas odpowiedzi.
"""

from __future__ import annotations

from typing import Any

from brain.llm import OllamaClient
from brain.personality import build_context_message, build_system_prompt
from config import Settings, UserSettings


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


class FakeMessage:
    """Minimalna wiadomość historii — tyle, ile potrzebuje ``_payload``."""

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def to_ollama(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


# --------------------------------------------------------------------------- #
# Prompt systemowy musi być identyczny między turami
# --------------------------------------------------------------------------- #


def test_prompt_systemowy_nie_zmienia_sie_miedzy_turami() -> None:
    """Dwie tury, ten sam użytkownik — prompt systemowy co do znaku ten sam.

    Gdyby wróciła do niego godzina albo blok wspomnień, serwer modelu liczyłby
    schematy narzędzi od nowa przy każdej wypowiedzi.
    """
    user = UserSettings(assistant_name="Miku", personality_traits="mówi krótko")

    first = build_system_prompt(user, language="pl", tool_rules="REGUŁY NARZĘDZI")
    second = build_system_prompt(user, language="pl", tool_rules="REGUŁY NARZĘDZI")

    assert first == second


def test_prompt_systemowy_nie_zawiera_godziny() -> None:
    """Znacznik czasu zmienia się co minutę — w prompcie systemowym to trucizna."""
    prompt = build_system_prompt(UserSettings(), language="pl")

    assert "Aktualna data i godzina" not in prompt
    assert "current date and time" not in prompt


def test_zmienne_tresci_maja_wlasna_wiadomosc() -> None:
    """Godzina, wspomnienia i podpowiedź o świeżych danych — wszystko poza promptem."""
    context = build_context_message(
        language="pl",
        extra_context="ZAPAMIĘTANE: użytkownik ma na imię Mariusz",
        request_hint="TO PYTANIE WYMAGA AKTUALNYCH DANYCH",
    )

    assert "Mariusz" in context
    assert "TO PYTANIE WYMAGA AKTUALNYCH DANYCH" in context
    assert "Aktualna data i godzina" in context
    # Nagłówek jest konieczny: wiadomość jedzie z rolą „user", więc bez niego
    # model mógłby odpowiedzieć NA kontekst zamiast na pytanie.
    assert context.startswith("[KONTEKST")


def test_pusty_kontekst_to_pusty_lancuch() -> None:
    """Bez wspomnień i bez podpowiedzi nie wysyłamy samego nagłówka."""
    assert build_context_message(language="pl", include_time=False) == ""


# --------------------------------------------------------------------------- #
# Kształt żądania do Ollamy
# --------------------------------------------------------------------------- #


def payload_for(context: str, *, tools: bool = True) -> dict[str, Any]:
    client = OllamaClient(make_settings())
    schemas = [{"type": "function", "function": {"name": "test.echo"}}] if tools else None
    return client._payload(
        [FakeMessage("user", "pytanie użytkownika")],
        stream=True,
        system="STAŁY PROMPT",
        tools=schemas,
        context=context,
    )


def test_kontekst_jedzie_jako_wiadomosc_uzytkownika() -> None:
    """Rola „system" na końcu kosztowała 40 s na turę — patrz nagłówek modułu."""
    payload = payload_for("[KONTEKST] godzina 21:37")
    roles = [message["role"] for message in payload["messages"]]

    assert roles == ["system", "user", "user"]
    assert payload["messages"][-1]["content"].startswith("[KONTEKST]")


def test_bez_kontekstu_nie_ma_dodatkowej_wiadomosci() -> None:
    payload = payload_for("")
    assert [message["role"] for message in payload["messages"]] == ["system", "user"]


def test_prompt_systemowy_zostaje_pierwszy() -> None:
    """Narzędzia renderują się razem z blokiem systemowym — musi być na początku."""
    payload = payload_for("[KONTEKST] cokolwiek")

    assert payload["messages"][0] == {"role": "system", "content": "STAŁY PROMPT"}
    assert payload["tools"]


def test_ten_sam_prompt_daje_ten_sam_prefiks() -> None:
    """Dwa kolejne żądania różnią się WYŁĄCZNIE ostatnią wiadomością."""
    first = payload_for("[KONTEKST] godzina 21:37")
    second = payload_for("[KONTEKST] godzina 21:38")

    assert first["messages"][:-1] == second["messages"][:-1]
    assert first["tools"] == second["tools"]
