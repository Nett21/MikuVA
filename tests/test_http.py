"""Testy kontrolowanego dostępu do sieci (Faza 9).

**Żaden test nie wychodzi do internetu.** Ruch jest podstawiany
``httpx.MockTransport`` — tym samym mechanizmem, którego używa biblioteka do
własnych testów — więc sprawdzamy nasz kod, a nie cudzy serwer i nie łącze.

Najważniejsza grupa: co jest odrzucane **przed** wysłaniem żądania. Bez tych
blokad model namówiony treścią strony mógłby sięgnąć po własną Ollamę
(``127.0.0.1:11434``), po metadane maszyny w chmurze (``169.254.169.254``) albo po
router w sieci domowej.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import pytest

from config import Settings
from host.http import (
    HttpPolicy,
    NetworkError,
    UrlRefusedError,
    build_headers,
    check_url,
    fetch,
    fetch_json,
    hard_offline,
    network_available,
    redact,
    redact_url,
    secret_value,
)


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def policy(**overrides: Any) -> HttpPolicy:
    values: dict[str, Any] = {"timeout_s": 5.0, "max_bytes": 10_000, "max_redirects": 2}
    values.update(overrides)
    return HttpPolicy(**values)


def client_for(handler: Any) -> httpx.AsyncClient:
    """Klient z podstawionym transportem — nic nie opuszcza procesu."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)


def text_response(
    body: str,
    *,
    status: int = 200,
    content_type: str = "text/html; charset=utf-8",
    **headers: str,
) -> httpx.Response:
    return httpx.Response(
        status, text=body, headers={"content-type": content_type, **headers}
    )


@pytest.fixture(autouse=True)
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nazwy domen rozwiązujemy na ustalony adres publiczny — bez pytania DNS-u.

    Testy nie mogą zależeć od tego, czy maszyna ma internet ani co odpowie serwer
    nazw. Adresy prywatne w testach podajemy wprost, więc ta atrapa ich nie dotyczy.
    """
    import socket

    import host.http

    def resolve(host: str, *args: Any, **kwargs: Any) -> list[Any]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(host.http.socket, "getaddrinfo", resolve)


# --------------------------------------------------------------------------- #
# Co jest odrzucane przed wysłaniem żądania
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "adres",
    [
        "file:///etc/passwd",
        "ftp://serwer/plik",
        "gopher://staroc",
        "javascript:alert(1)",
        "data:text/html,<h1>x</h1>",
    ],
)
def test_tylko_http_i_https(adres: str) -> None:
    with pytest.raises(UrlRefusedError) as blad:
        check_url(adres, policy())
    assert "http" in blad.value.message


@pytest.mark.parametrize(
    "adres",
    [
        "http://127.0.0.1:11434/api/chat",   # własna Ollama
        "http://localhost:8080/",
        "http://[::1]/",
        "http://192.168.0.1/admin",           # router w sieci domowej
        "http://10.0.0.5/",
        "http://172.16.3.4/",
        "http://169.254.169.254/latest/meta-data/",  # metadane maszyny w chmurze
        "http://0.0.0.0/",
        "https://router.local/",
        "https://cos.internal/dane",
    ],
)
def test_adresy_prywatne_i_lokalne_sa_odrzucane(adres: str) -> None:
    """To jest sedno ochrony: model nie może sięgnąć do tej maszyny ani do LAN-u."""
    with pytest.raises(UrlRefusedError) as blad:
        check_url(adres, policy())
    # Sprawdzamy SEDNO odmowy, nie brzmienie zdania: komunikat idzie przez
    # i18n i zależy od UI_LANGUAGE, a to, że adres został odrzucony — nie.
    assert "local" in blad.value.message or "not a public" in blad.value.message


def test_nazwa_wskazujaca_na_adres_prywatny_tez_jest_odrzucana(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sprawdzenie po rozwiązaniu nazwy — bez tego blokada byłaby ozdobą."""
    import socket

    import host.http

    monkeypatch.setattr(
        host.http.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.50", 0))
        ],
    )
    with pytest.raises(UrlRefusedError) as blad:
        check_url("https://moja-domena.example/panel", policy())
    assert "192.168.1.50" in blad.value.message


def test_wlasna_instancja_w_sieci_lokalnej_da_sie_dopuscic_swiadomie() -> None:
    """Jedyny realny przypadek: własny SearXNG pod 192.168.x.x."""
    assert check_url("http://192.168.1.10:8888/search", policy(allow_private_hosts=True))


def test_adres_z_loginem_i_haslem_jest_odrzucany() -> None:
    with pytest.raises(UrlRefusedError) as blad:
        check_url("https://user:tajne@example.org/", policy())
    assert "password" in blad.value.message


@pytest.mark.parametrize("port", [22, 25, 3306, 6379, 27017])
def test_porty_uslug_innych_niz_www_sa_odrzucane(port: int) -> None:
    with pytest.raises(UrlRefusedError):
        check_url(f"http://example.org:{port}/", policy())


def test_brak_schematu_dopelniamy_https_a_nie_http() -> None:
    assert check_url("example.org/artykul", policy()).startswith("https://")


def test_pusty_adres_daje_czytelny_blad() -> None:
    with pytest.raises(UrlRefusedError) as blad:
        check_url("   ", policy())
    assert "nie podano adresu" in blad.value.message


# --------------------------------------------------------------------------- #
# Tryb offline
# --------------------------------------------------------------------------- #


def test_offline_on_wylacza_narzedzia_sieciowe() -> None:
    usable, reason = network_available(make_settings(offline_mode="on"))
    assert not usable and "offline" in reason
    assert hard_offline(make_settings(offline_mode="on"))


def test_tryb_auto_nie_odbiera_dostepu_do_sieci() -> None:
    """``auto`` mówi „nie pobieraj modeli", a nie „nie wolno zapytać o pogodę".

    Regresja z projektu: przy pierwszym podejściu ``auto`` (rozstrzygnięty jako
    offline, bo model Whispera leżał na dysku) wyłączał WSZYSTKIE narzędzia
    sieciowe — samo posiadanie modelu na dysku odbierało asystentowi internet.
    """
    settings = make_settings(offline_mode="auto")
    usable, reason = network_available(settings)
    assert usable, reason
    assert not hard_offline(settings)


def test_wylaczenie_web_enabled_tez_wylacza() -> None:
    usable, reason = network_available(make_settings(web_enabled=False))
    assert not usable and "WEB_ENABLED" in reason


def test_pobieranie_w_trybie_offline_nie_wysyla_zadania() -> None:
    wywolania: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - nie powinno
        wywolania.append(str(request.url))
        return text_response("nie powinno się zdarzyć")

    with pytest.raises(NetworkError) as blad:
        run(
            fetch(
                "https://example.org/",
                policy=policy(offline=True),
                client=client_for(handler),
            )
        )
    assert blad.value.offline and wywolania == []


# --------------------------------------------------------------------------- #
# Sekrety
# --------------------------------------------------------------------------- #


def test_klucz_api_jest_zamazywany_w_adresie() -> None:
    url = "https://api.example.org/v1/search?q=rower&key=SEKRETNY-KLUCZ-123&page=2"
    zamazany = redact_url(url)

    assert "SEKRETNY-KLUCZ-123" not in zamazany
    assert "q=rower" in zamazany and "page=2" in zamazany


@pytest.mark.parametrize(
    "tekst",
    [
        "GET https://api.example.org/x?api_key=abc123def",
        "błąd: token=abc123def wygasł",
        "Authorization: Bearer abc123def",
        "https://user:abc123def@example.org/",
    ],
)
def test_redakcja_usuwa_sekrety_z_tekstu(tekst: str) -> None:
    assert "abc123def" not in redact(tekst)


def test_secret_value_dziala_dla_secretstr_i_tekstu() -> None:
    from pydantic import SecretStr

    assert secret_value(SecretStr("tajne")) == "tajne"
    assert secret_value("tajne") == "tajne"
    assert secret_value(None) == ""


def test_klucz_nie_trafia_do_logu_przy_bledzie(caplog: pytest.LogCaptureFixture) -> None:
    """Najważniejszy test tej grupy: log z błędem nie może zawierać klucza."""

    def handler(request: httpx.Request) -> httpx.Response:
        return text_response("brak dostępu", status=403)

    from logging_setup import SecretRedactingFilter

    caplog.handler.addFilter(SecretRedactingFilter())
    with caplog.at_level(logging.DEBUG), pytest.raises(NetworkError) as blad:
        run(
            fetch(
                "https://api.example.org/v1/data",
                policy=policy(),
                params={"key": "SEKRETNY-KLUCZ-123", "q": "rower"},
                client=client_for(handler),
            )
        )

    assert "SEKRETNY-KLUCZ-123" not in blad.value.user_message
    assert "SEKRETNY-KLUCZ-123" not in caplog.text
    assert "***" in blad.value.message


# --------------------------------------------------------------------------- #
# Pobieranie: limity i błędy
# --------------------------------------------------------------------------- #


def test_pobranie_strony_zwraca_tekst() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"].startswith("miku-assistant")
        return text_response("<h1>Cześć</h1>")

    response = run(fetch("https://example.org/", policy=policy(), client=client_for(handler)))

    assert response.ok and "Cześć" in response.text
    assert response.content_type == "text/html"


def test_zbyt_duza_odpowiedz_jest_obcinana() -> None:
    """Serwer bez ``content-length`` (chunked) — obcinamy przy czytaniu."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=httpx.ByteStream(b"x" * 50_000),
            headers={"content-type": "text/plain"},
        )

    response = run(
        fetch("https://example.org/", policy=policy(max_bytes=1_000), client=client_for(handler))
    )

    assert response.truncated and len(response.text) <= 1_000


def test_zadeklarowany_rozmiar_ponad_limit_jest_odrzucany_bez_pobierania() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return text_response("x", **{"content-length": "9999999"})

    with pytest.raises(NetworkError) as blad:
        run(
            fetch(
                "https://example.org/",
                policy=policy(max_bytes=1_000),
                client=client_for(handler),
            )
        )
    assert "limit" in blad.value.message


def test_tresc_binarna_nie_jest_pobierana() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\x00\x01", headers={"content-type": "image/png"})

    with pytest.raises(NetworkError) as blad:
        run(fetch("https://example.org/obraz.png", policy=policy(), client=client_for(handler)))
    assert "not text" in blad.value.message


@pytest.mark.parametrize(
    ("status", "fragment"),
    [(404, "nie ma pod tym adresem"), (403, "odmówił dostępu"), (429, "przerwę"), (500, "serwera")],
)
def test_kody_bledow_maja_zrozumiale_podpowiedzi(status: int, fragment: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return text_response("błąd", status=status)

    with pytest.raises(NetworkError) as blad:
        run(fetch("https://example.org/", policy=policy(), client=client_for(handler)))
    assert fragment in blad.value.user_message


def test_przekroczony_czas_daje_komunikat_a_nie_wyjatek_biblioteki() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("za długo")

    with pytest.raises(NetworkError) as blad:
        run(fetch("https://example.org/", policy=policy(), client=client_for(handler)))
    assert "did not answer" in blad.value.message


def test_brak_polaczenia_daje_komunikat_z_podpowiedzia() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("brak trasy do hosta")

    with pytest.raises(NetworkError) as blad:
        run(fetch("https://example.org/", policy=policy(), client=client_for(handler)))
    assert "connect" in blad.value.message
    assert "internet" in blad.value.hint


def test_brak_odpowiedzi_dns_jest_bledem_sieci_nie_adresu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket

    import host.http

    def failing(*args: Any, **kwargs: Any) -> list[Any]:
        raise socket.gaierror("nie znaleziono")

    monkeypatch.setattr(host.http.socket, "getaddrinfo", failing)
    with pytest.raises(NetworkError) as blad:
        check_url("https://nie-ma-takiej-domeny.example/", policy())
    assert "resolve the name" in blad.value.message


# --------------------------------------------------------------------------- #
# Przekierowania
# --------------------------------------------------------------------------- #


def test_przekierowanie_jest_sprawdzane_od_nowa() -> None:
    """Serwer nie może przekierować nas na adres prywatny po zaakceptowaniu publicznego."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "example.org":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/token"})
        return text_response("sekret chmury")  # pragma: no cover - nie powinno się zdarzyć

    with pytest.raises(UrlRefusedError) as blad:
        run(fetch("https://example.org/", policy=policy(), client=client_for(handler)))
    assert "not a public" in blad.value.message


def test_przekierowanie_na_adres_publiczny_jest_podejmowane() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(301, headers={"location": "https://example.org/cel"})
        return text_response("<p>docelowa treść</p>")

    response = run(
        fetch("https://example.org/start", policy=policy(), client=client_for(handler))
    )
    assert "docelowa treść" in response.text
    assert response.url.endswith("/cel")


def test_petla_przekierowan_ma_koniec() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://example.org/kolo"})

    with pytest.raises(NetworkError) as blad:
        run(
            fetch(
                "https://example.org/kolo",
                policy=policy(max_redirects=2),
                client=client_for(handler),
            )
        )
    assert "redirect" in blad.value.message


# --------------------------------------------------------------------------- #
# JSON i nagłówki
# --------------------------------------------------------------------------- #


def test_fetch_json_zwraca_dane() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"wynik": 42}, headers={"content-type": "application/json"})

    assert run(
        fetch_json("https://api.example.org/", policy=policy(), client=client_for(handler))
    ) == {"wynik": 42}


def test_nie_json_w_odpowiedzi_daje_czytelny_blad() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return text_response("<html>strona błędu</html>", content_type="application/json")

    with pytest.raises(NetworkError) as blad:
        run(fetch_json("https://api.example.org/", policy=policy(), client=client_for(handler)))
    assert "JSON" in blad.value.message


def test_naglowki_nie_zawieraja_ciasteczek_ani_autoryzacji() -> None:
    headers = build_headers(policy(user_agent="atrapa/1.0"), accept="text/html")
    assert headers["User-Agent"] == "atrapa/1.0"
    assert "Cookie" not in headers and "Authorization" not in headers


def test_kodowanie_z_naglowka_ma_pierwszenstwo_nad_domyslnym() -> None:
    """Ta sama strona na dwóch maszynach musi dać ten sam tekst."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content="zażółć gęślą jaźń".encode("iso-8859-2"),
            headers={"content-type": "text/html; charset=iso-8859-2"},
        )

    response = run(fetch("https://example.org/", policy=policy(), client=client_for(handler)))
    assert "zażółć gęślą jaźń" in response.text
