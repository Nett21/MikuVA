"""Testy narzędzi sieciowych: web, pogoda, wiadomości, YouTube (Faza 9).

**Żaden test nie wychodzi do internetu i żaden nie potrzebuje klucza API.**
Odpowiedzi serwerów podstawia ``httpx.MockTransport``, a klucz w testach to
zmyślony łańcuch — sprawdzamy też, czy nie wycieka do logu ani do wyniku.

Grupa najważniejsza dla tej fazy: **co się dzieje, gdy sieci nie ma**. Narzędzie
ma wtedy zwrócić czytelny komunikat, a nie wyjątek — asystent działa dalej,
tylko bez tego jednego narzędzia.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import pytest

from config import Settings
from security.risk import RiskLevel
from tools.base import ToolContext, ToolError
from tools.news import HeadlinesArgs, NewsSearchArgs, build_news_tools
from tools.weather import ForecastArgs, WeatherArgs, build_weather_tools, describe_code
from tools.web import FetchArgs, SearchArgs, build_web_tools
from tools.webtext import ContentError, extract_readable, parse_feed, strip_tags
from tools.youtube import PlayArgs, TranscriptArgs, build_youtube_tools, video_id
from tools.youtube import SearchArgs as YouTubeSearchArgs


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {"offline_mode": "off"}
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def ctx(language: str = "pl") -> ToolContext:
    return ToolContext(settings=make_settings(), language=language)


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def client_for(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)


def json_response(payload: Any) -> httpx.Response:
    return httpx.Response(200, json=payload, headers={"content-type": "application/json"})


def html_response(body: str) -> httpx.Response:
    return httpx.Response(200, text=body, headers={"content-type": "text/html; charset=utf-8"})


def xml_response(body: str) -> httpx.Response:
    return httpx.Response(200, text=body, headers={"content-type": "application/rss+xml"})


@pytest.fixture(autouse=True)
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nazwy rozwiązujemy na ustalony adres publiczny — bez pytania DNS-u."""
    import socket

    import host.http

    monkeypatch.setattr(
        host.http.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))
        ],
    )


def tools_from(builder: Any, handler: Any, **overrides: Any) -> dict[str, Any]:
    settings = make_settings(**overrides)
    return {tool.spec.name: tool for tool in builder(settings, client=client_for(handler))}


# --------------------------------------------------------------------------- #
# Poziomy ryzyka i dostępność
# --------------------------------------------------------------------------- #


def test_narzedzia_sieciowe_sa_medium_a_odtwarzanie_high() -> None:
    """Ruch wychodzący to MEDIUM; zajęcie ekranu użytkownika to HIGH."""
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        return json_response({})

    web = tools_from(build_web_tools, handler)
    weather = tools_from(build_weather_tools, handler)
    news = tools_from(build_news_tools, handler)
    youtube = tools_from(build_youtube_tools, handler)

    for name in ("web.search", "web.fetch"):
        assert web[name].spec.risk is RiskLevel.MEDIUM
        assert web[name].spec.requires_network
    assert weather["weather.current"].spec.risk is RiskLevel.MEDIUM
    assert news["news.search"].spec.risk is RiskLevel.MEDIUM
    assert youtube["youtube.transcript"].spec.risk is RiskLevel.MEDIUM
    assert youtube["youtube.play"].spec.risk is RiskLevel.HIGH


def test_tryb_offline_wylacza_wszystkie_narzedzia_sieciowe() -> None:
    """Program działa dalej — tylko te narzędzia są niewidoczne dla modelu."""
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        return json_response({})

    for builder in (build_web_tools, build_weather_tools, build_news_tools, build_youtube_tools):
        narzedzia = tools_from(builder, handler, offline_mode="on")
        for tool in narzedzia.values():
            usable, reason = tool.available()
            assert not usable and "offline" in reason


def test_wylaczenie_wyszukiwania_ukrywa_tylko_web_search() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        return json_response({})

    narzedzia = tools_from(build_web_tools, handler, search_provider="none")
    assert not narzedzia["web.search"].available()[0]
    assert narzedzia["web.fetch"].available()[0]


def test_searxng_bez_adresu_instancji_jest_niedostepne() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        return json_response({})

    narzedzia = tools_from(build_web_tools, handler, search_provider="searxng")
    usable, reason = narzedzia["web.search"].available()
    assert not usable and "SEARCH_BASE_URL" in reason


# --------------------------------------------------------------------------- #
# web.search
# --------------------------------------------------------------------------- #


_DDG_PAGE = """
<html><body>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Frower.example%2Ftest">
    Rower górski <b>Kellys</b>
  </a>
  <a class="result__snippet">Opis <b>roweru</b> Kellys w sklepie</a>
</div>
<div class="result">
  <a class="result__a" href="https://drugi.example/artykul">Drugi wynik</a>
  <a class="result__snippet">Krótki opis drugiego wyniku</a>
</div>
</body></html>
"""


def test_wyszukiwanie_zwraca_tytuly_adresy_i_opisy() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "html.duckduckgo.com"
        assert request.url.params["q"] == "rower kellys"
        return html_response(_DDG_PAGE)

    narzedzia = tools_from(build_web_tools, handler)
    wynik = run(narzedzia["web.search"].run(SearchArgs(query="rower kellys"), ctx()))

    assert wynik.ok and wynik.data["count"] == 2
    first = wynik.data["results"][0]
    # Przekierowanie wyszukiwarki jest rozpakowane do prawdziwego adresu.
    assert first["url"] == "https://rower.example/test"
    assert first["title"] == "Rower górski Kellys"
    assert "Kellys" in first["snippet"]
    # Treść z internetu = dane niezaufane (bariera przed kolejnym wywołaniem).
    assert wynik.untrusted is True


def test_brak_wynikow_daje_zrozumialy_komunikat() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return html_response("<html><body>nic tu nie ma</body></html>")

    narzedzia = tools_from(build_web_tools, handler)
    with pytest.raises(ToolError) as blad:
        run(narzedzia["web.search"].run(SearchArgs(query="cokolwiek"), ctx()))
    assert "no results" in blad.value.message


def test_wyszukiwanie_przez_searxng_czyta_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["format"] == "json"
        return json_response(
            {"results": [{"title": "Wynik", "url": "https://a.example/x", "content": "opis"}]}
        )

    narzedzia = tools_from(
        build_web_tools,
        handler,
        search_provider="searxng",
        search_base_url="https://searx.example",
    )
    wynik = run(narzedzia["web.search"].run(SearchArgs(query="test"), ctx()))
    assert wynik.data["results"][0]["url"] == "https://a.example/x"
    assert wynik.data["provider"] == "searxng"


def test_blad_sieci_w_wyszukiwaniu_nie_wywala_asystenta() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("brak trasy do hosta")

    narzedzia = tools_from(build_web_tools, handler)
    with pytest.raises(ToolError) as blad:
        run(narzedzia["web.search"].run(SearchArgs(query="rower"), ctx()))

    # Komunikat mówi, co się stało i co z tym zrobić — to on trafia do modelu,
    # a potem do użytkownika (także na głos).
    assert "connect" in blad.value.message and "internet" in blad.value.message


# --------------------------------------------------------------------------- #
# web.fetch
# --------------------------------------------------------------------------- #


_ARTICLE = """
<html><head><title>Rower Kellys — test</title>
<meta name="description" content="Krótki opis testu"></head>
<body>
<nav>Menu: strona główna, kontakt</nav>
<script>console.log("nie chcę tego widzieć")</script>
<style>body { color: red }</style>
<h1>Rower Kellys</h1>
<p>Pierwszy akapit o rowerze.</p>
<p>Drugi akapit o hamulcach tarczowych.</p>
<footer>Stopka z prawami autorskimi</footer>
</body></html>
"""


def test_pobranie_strony_zwraca_tytul_i_tresc_bez_kodu() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return html_response(_ARTICLE)

    narzedzia = tools_from(build_web_tools, handler)
    wynik = run(narzedzia["web.fetch"].run(FetchArgs(url="https://rower.example/test"), ctx()))

    assert wynik.data["title"] == "Rower Kellys — test"
    assert "Pierwszy akapit" in wynik.data["text"]
    assert "console.log" not in wynik.data["text"]  # skrypt wyrzucony
    assert "color: red" not in wynik.data["text"]  # styl wyrzucony
    assert wynik.untrusted is True


def test_focus_zostawia_tylko_akapity_z_fraza() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return html_response(_ARTICLE)

    narzedzia = tools_from(build_web_tools, handler)
    wynik = run(
        narzedzia["web.fetch"].run(
            FetchArgs(url="https://rower.example/test", focus="hamulcach"), ctx()
        )
    )
    assert "hamulcach" in wynik.data["text"]
    assert "Pierwszy akapit" not in wynik.data["text"]


def test_pobranie_adresu_prywatnego_jest_odrzucane() -> None:
    """Model nie sięgnie po własną Ollamę ani po metadane maszyny w chmurze."""
    wywolania: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        wywolania.append(str(request.url))
        return html_response("sekret")

    narzedzia = tools_from(build_web_tools, handler)
    for adres in ("http://127.0.0.1:11434/api/tags", "http://169.254.169.254/latest/meta-data/"):
        with pytest.raises(ToolError):
            run(narzedzia["web.fetch"].run(FetchArgs(url=adres), ctx()))
    assert wywolania == []  # żadne żądanie nie zostało wysłane


def test_strona_bez_tekstu_mowi_o_javascripcie() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return html_response("<html><body><div id='app'></div></body></html>")

    narzedzia = tools_from(build_web_tools, handler)
    with pytest.raises(ToolError) as blad:
        run(narzedzia["web.fetch"].run(FetchArgs(url="https://app.example/"), ctx()))
    assert "JavaScript" in blad.value.message


def test_json_z_api_jest_zwracany_jako_tekst() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"cena": 1999, "waluta": "PLN"})

    narzedzia = tools_from(build_web_tools, handler)
    wynik = run(narzedzia["web.fetch"].run(FetchArgs(url="https://api.example/cena"), ctx()))
    assert '"cena"' in wynik.data["text"]


# --------------------------------------------------------------------------- #
# Pogoda
# --------------------------------------------------------------------------- #


def _weather_handler(request: httpx.Request) -> httpx.Response:
    host = request.url.host
    if host.startswith("geocoding-api"):
        assert request.url.params["name"] == "Wrocław"
        return json_response(
            {
                "results": [
                    {
                        "name": "Wrocław",
                        "country": "Polska",
                        "latitude": 51.1,
                        "longitude": 17.03,
                        "timezone": "Europe/Warsaw",
                    }
                ]
            }
        )
    if "current" in request.url.params:
        return json_response(
            {
                "current": {
                    "time": "2026-08-17T16:00",
                    "temperature_2m": 24.3,
                    "apparent_temperature": 25.1,
                    "relative_humidity_2m": 52,
                    "precipitation": 0.0,
                    "weather_code": 2,
                    "wind_speed_10m": 12.4,
                }
            }
        )
    return json_response(
        {
            "daily": {
                "time": ["2026-08-17", "2026-08-18"],
                "weather_code": [3, 61],
                "temperature_2m_max": [25.0, 21.0],
                "temperature_2m_min": [14.0, 13.0],
                "precipitation_sum": [0.0, 4.2],
                "wind_speed_10m_max": [18.0, 22.0],
            }
        }
    )


def test_pogoda_teraz_bez_klucza_api() -> None:
    """Open-Meteo nie wymaga rejestracji — działa na świeżej instalacji."""
    narzedzia = tools_from(build_weather_tools, _weather_handler)
    wynik = run(narzedzia["weather.current"].run(WeatherArgs(location="Wrocław"), ctx()))

    assert wynik.ok
    assert wynik.data["location"] == "Wrocław, Polska"
    assert wynik.data["temperature"] == 24.3
    assert wynik.data["description"] == "częściowe zachmurzenie"
    assert wynik.data["units"]["temperature"] == "°C"
    assert "24.3°C" in wynik.display


def test_opisy_pogody_nie_zaleza_od_locale_maszyny() -> None:
    """Kody WMO tłumaczymy w kodzie — nie przez ustawienia regionalne systemu."""
    assert describe_code(61, language="pl") == "słaby deszcz"
    assert describe_code(61, language="en") == "light rain"
    assert describe_code(999, language="pl") == "brak opisu"


def test_jednostki_imperialne_sa_respektowane() -> None:
    narzedzia = tools_from(build_weather_tools, _weather_handler, weather_units="imperial")
    wynik = run(narzedzia["weather.current"].run(WeatherArgs(location="Wrocław"), ctx()))
    assert wynik.data["units"] == {"temperature": "°F", "wind": "mph"}


def test_prognoza_na_kilka_dni() -> None:
    narzedzia = tools_from(build_weather_tools, _weather_handler)
    wynik = run(
        narzedzia["weather.forecast"].run(ForecastArgs(location="Wrocław", days=2), ctx())
    )

    assert len(wynik.data["days"]) == 2
    assert wynik.data["days"][1]["description"] == "słaby deszcz"
    assert wynik.data["days"][0]["max"] == 25.0


def test_brak_miejsca_i_brak_domyslnego_daje_prosbe_o_uzupelnienie() -> None:
    narzedzia = tools_from(build_weather_tools, _weather_handler)
    with pytest.raises(ToolError) as blad:
        run(narzedzia["weather.current"].run(WeatherArgs(), ctx()))
    assert "WEATHER_DEFAULT_LOCATION" in blad.value.message


def test_domyslne_miejsce_z_konfiguracji_jest_uzywane() -> None:
    narzedzia = tools_from(
        build_weather_tools, _weather_handler, weather_default_location="Wrocław"
    )
    wynik = run(narzedzia["weather.current"].run(WeatherArgs(), ctx()))
    assert wynik.data["location"].startswith("Wrocław")


def test_nieznane_miejsce_daje_czytelny_blad() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"results": []})

    narzedzia = tools_from(build_weather_tools, handler)
    with pytest.raises(ToolError) as blad:
        run(narzedzia["weather.current"].run(WeatherArgs(location="Xyzabc"), ctx()))
    assert "could not find a place" in blad.value.message


def test_padniety_serwis_pogodowy_nie_zatrzymuje_asystenta() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="chwilowo niedostępne")

    narzedzia = tools_from(build_weather_tools, handler)
    with pytest.raises(ToolError) as blad:
        run(narzedzia["weather.current"].run(WeatherArgs(location="Wrocław"), ctx()))
    assert "503" in blad.value.message and "serwera" in blad.value.message


# --------------------------------------------------------------------------- #
# Wiadomości
# --------------------------------------------------------------------------- #


_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Kanał testowy</title>
  <item>
    <title>Pierwsza wiadomość</title>
    <link>https://news.example/1</link>
    <pubDate>Mon, 17 Aug 2026 14:00:00 GMT</pubDate>
    <description>&lt;p&gt;Opis pierwszej&lt;/p&gt;</description>
  </item>
  <item>
    <title>Druga wiadomość</title>
    <link>https://news.example/2</link>
    <pubDate>Mon, 17 Aug 2026 12:30:00 GMT</pubDate>
    <description>Opis drugiej</description>
  </item>
</channel></rss>
"""

_ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Wpis Atom</title>
    <link href="https://atom.example/wpis"/>
    <updated>2026-08-17T10:00:00Z</updated>
    <summary>Streszczenie wpisu</summary>
  </entry>
</feed>
"""


def test_naglowki_z_kanalu_rss() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return xml_response(_RSS)

    narzedzia = tools_from(
        build_news_tools, handler, news_feeds="https://news.example/rss"
    )
    wynik = run(narzedzia["news.headlines"].run(HeadlinesArgs(limit=2), ctx()))

    assert wynik.data["count"] == 2
    first = wynik.data["items"][0]
    assert first["title"] == "Pierwsza wiadomość"
    assert first["link"] == "https://news.example/1"
    assert first["published"].startswith("2026-08-17T14:00")
    assert "Opis pierwszej" in first["summary"]  # znaczniki HTML zdjęte
    assert wynik.untrusted is True


def test_kanal_atom_tez_dziala() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return xml_response(_ATOM)

    items = parse_feed(_ATOM, limit=5, source="atom.example")
    assert items[0].title == "Wpis Atom" and items[0].link == "https://atom.example/wpis"
    del handler


def test_brak_kanalow_wylacza_naglowki_ale_nie_szukanie() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        return xml_response(_RSS)

    narzedzia = tools_from(build_news_tools, handler)
    usable, reason = narzedzia["news.headlines"].available()
    assert not usable and "NEWS_FEEDS" in reason
    assert narzedzia["news.search"].available()[0]


def test_jeden_padniety_kanal_nie_zabiera_pozostalych() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "zepsuty" in str(request.url):
            return httpx.Response(500, text="błąd")
        return xml_response(_RSS)

    narzedzia = tools_from(
        build_news_tools,
        handler,
        news_feeds="https://zepsuty.example/rss,https://news.example/rss",
    )
    wynik = run(narzedzia["news.headlines"].run(HeadlinesArgs(limit=4), ctx()))

    assert wynik.data["count"] >= 1
    assert "problems" in wynik.data  # o padniętym kanale mówimy wprost


def test_szukanie_wiadomosci_podstawia_zapytanie_w_szablonie() -> None:
    widziane: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        widziane.append(str(request.url))
        return xml_response(_RSS)

    narzedzia = tools_from(
        build_news_tools,
        handler,
        news_search_url="https://news.example/rss/search?q={query}&hl={language}",
    )
    wynik = run(narzedzia["news.search"].run(NewsSearchArgs(query="rower miejski"), ctx("pl")))

    assert "q=rower+miejski" in widziane[0] and "hl=pl" in widziane[0]
    assert wynik.data["count"] == 2


def test_kanal_z_deklaracja_encji_jest_odrzucany() -> None:
    """Ochrona przed „billion laughs" — prawdziwy kanał nie potrzebuje encji."""
    bomba = (
        '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">]>'
        "<rss><channel><item><title>&lol;</title></item></channel></rss>"
    )
    with pytest.raises(ContentError) as blad:
        parse_feed(bomba)
    assert "XML entity" in blad.value.message


def test_niepoprawny_xml_daje_zrozumialy_blad() -> None:
    with pytest.raises(ContentError) as blad:
        parse_feed("<rss><channel><item>bez zamknięcia")
    assert "XML" in blad.value.message


# --------------------------------------------------------------------------- #
# YouTube
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "zapis",
    [
        "dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ&t=42",
        "youtube.com/embed/dQw4w9WgXcQ",
    ],
)
def test_identyfikator_filmu_z_roznych_zapisow(zapis: str) -> None:
    assert video_id(zapis) == "dQw4w9WgXcQ"


def test_adres_z_innego_serwisu_jest_odrzucany() -> None:
    with pytest.raises(ToolError) as blad:
        video_id("https://vimeo.com/12345678901")
    assert "nie YouTube" in blad.value.message


def test_szukanie_na_youtube_wymaga_klucza() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        return json_response({})

    narzedzia = tools_from(build_youtube_tools, handler)
    usable, reason = narzedzia["youtube.search"].available()
    assert not usable
    assert "YOUTUBE_API_KEY" in reason and "web.search" in reason


def test_szukanie_z_kluczem_zwraca_filmy_a_klucz_nie_wycieka(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from logging_setup import SecretRedactingFilter

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["key"] == "TAJNY-KLUCZ-YT"
        return json_response(
            {
                "items": [
                    {
                        "id": {"videoId": "dQw4w9WgXcQ"},
                        "snippet": {
                            "title": "Testowy film",
                            "channelTitle": "Kanał",
                            "publishedAt": "2026-08-01T10:00:00Z",
                        },
                    }
                ]
            }
        )

    caplog.handler.addFilter(SecretRedactingFilter())
    narzedzia = tools_from(build_youtube_tools, handler, youtube_api_key="TAJNY-KLUCZ-YT")
    with caplog.at_level(logging.DEBUG):
        wynik = run(
            narzedzia["youtube.search"].run(YouTubeSearchArgs(query="test"), ctx())
        )

    assert wynik.data["videos"][0]["url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    # Klucz poszedł w żądaniu, ale nie ma go ani w wyniku, ani w logu.
    assert "TAJNY-KLUCZ-YT" not in wynik.to_json()
    assert "TAJNY-KLUCZ-YT" not in caplog.text


def test_transkrypcja_dziala_bez_klucza() -> None:
    napisy = (
        '<?xml version="1.0" encoding="utf-8"?><transcript>'
        '<text start="0" dur="2">Cześć, tu pierwsza linia</text>'
        '<text start="2" dur="2">a to druga linia napisów</text>'
        "</transcript>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["v"] == "dQw4w9WgXcQ"
        return httpx.Response(200, text=napisy, headers={"content-type": "text/xml"})

    narzedzia = tools_from(build_youtube_tools, handler)
    wynik = run(
        narzedzia["youtube.transcript"].run(TranscriptArgs(video="dQw4w9WgXcQ"), ctx())
    )

    assert "pierwsza linia" in wynik.data["text"]
    assert "druga linia" in wynik.data["text"]
    assert wynik.untrusted is True


def test_film_bez_napisow_mowi_o_tym_wprost() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="", headers={"content-type": "text/xml"})

    narzedzia = tools_from(build_youtube_tools, handler)
    with pytest.raises(ToolError) as blad:
        run(narzedzia["youtube.transcript"].run(TranscriptArgs(video="dQw4w9WgXcQ"), ctx()))
    assert "no subtitles available" in blad.value.message


def test_odtworzenie_filmu_pyta_o_zgode_i_pokazuje_adres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import host.apps

    monkeypatch.setattr(host.apps, "has_graphical_session", lambda *args, **kwargs: True)

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        return json_response({})

    otwarte: list[list[str]] = []
    settings = make_settings()
    narzedzia = {
        tool.spec.name: tool
        for tool in build_youtube_tools(
            settings,
            client=client_for(handler),
            runner=lambda argv, **kwargs: otwarte.append([str(item) for item in argv]),
        )
    }
    play = narzedzia["youtube.play"]

    pytanie = play.confirmation(PlayArgs(video="dQw4w9WgXcQ"), language="pl")
    assert pytanie is not None and pytanie.risk is RiskLevel.HIGH
    assert "dQw4w9WgXcQ" in pytanie.summary
    assert "dźwięk" in "\n".join(pytanie.details)

    wynik = run(play.run(PlayArgs(video="dQw4w9WgXcQ"), ctx()))
    assert wynik.ok and otwarte  # otwarcie poszło przez atrapę, nie do przeglądarki


def test_odtwarzanie_bez_sesji_graficznej_jest_niedostepne(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.youtube

    monkeypatch.setattr(tools.youtube, "has_graphical_session", lambda *args, **kwargs: False)

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        return json_response({})

    narzedzia = tools_from(build_youtube_tools, handler)
    usable, reason = narzedzia["youtube.play"].available()
    assert not usable and "sesji graficznej" in reason


# --------------------------------------------------------------------------- #
# Przetwarzanie treści
# --------------------------------------------------------------------------- #


def test_wyciaganie_tekstu_pomija_nawigacje_i_stopke() -> None:
    article = extract_readable(_ARTICLE, max_chars=1_000)
    assert "Pierwszy akapit" in article.text
    assert "Menu: strona główna" not in article.text
    assert "Stopka" not in article.text


def test_strona_zlozona_z_samej_nawigacji_zwraca_cokolwiek() -> None:
    """Lepiej dać treść nawigacji niż pustkę — model sam oceni, czy to pomaga."""
    article = extract_readable(
        "<html><body><nav>Kontakt, regulamin, mapa strony</nav></body></html>"
    )
    assert "regulamin" in article.text


def test_dlugi_tekst_jest_obcinany_na_granicy_zdania() -> None:
    body = "<p>" + ("Zdanie o rowerze. " * 200) + "</p>"
    article = extract_readable(body, max_chars=300)
    assert article.truncated and len(article.text) < 400
    assert "obcięta" in article.text


def test_encje_html_sa_rozwijane() -> None:
    assert strip_tags("Rower &amp; hamulce &lt;test&gt;") == "Rower & hamulce <test>"
