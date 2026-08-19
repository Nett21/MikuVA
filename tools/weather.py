"""Pogoda — Open-Meteo, bez klucza API (Faza 9).

Oba narzędzia są MEDIUM: nic nie zmieniają, ale wysyłają zapytanie do serwisu
zewnętrznego (a razem z nim nazwę miejsca, o które pyta użytkownik).

Dlaczego **Open-Meteo**, a nie OpenWeatherMap czy podobne: nie wymaga klucza API
ani rejestracji, więc asystent po świeżej instalacji odpowiada na „jaka jest
pogoda?" bez żadnej konfiguracji. Klucz (``WEATHER_API_KEY``) jest przewidziany
w konfiguracji dla wariantów płatnych, ale **nie jest do niczego potrzebny** i
nigdzie nie trafia do logu.

Geokodowanie (nazwa miejsca → współrzędne) idzie do tego samego serwisu. Nie ma
tu żadnej wbudowanej listy miast: „Wrocław", „Warszawa" i „Reykjavík" przechodzą
tą samą drogą, a język wyników bierzemy z języka rozmowy.

Kody pogody (WMO) tłumaczymy **w kodzie**, po polsku i angielsku — nie przez
``locale`` maszyny. Ten sam kod ma znaczyć „zachmurzenie" niezależnie od tego,
jakie ustawienia regionalne ma komputer.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Final

import httpx
from pydantic import Field

from config import Settings, get_settings
from host.http import HttpPolicy, NetworkError, fetch_json, network_available
from security.risk import RiskLevel
from tools.base import BaseTool, Tool, ToolArgs, ToolContext, ToolError, ToolResult, ToolSpec

logger = logging.getLogger(__name__)

_GEOCODE_URL: Final[str] = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL: Final[str] = "https://api.open-meteo.com/v1/forecast"

# Kody pogody WMO. Pełna tabela ma kilkadziesiąt pozycji; grupujemy je tak, żeby
# opis dał się przeczytać na głos.
_WMO_PL: Final[dict[int, str]] = {
    0: "bezchmurnie", 1: "prawie bezchmurnie", 2: "częściowe zachmurzenie", 3: "zachmurzenie",
    45: "mgła", 48: "mgła osadzająca szron",
    51: "mżawka", 53: "mżawka", 55: "gęsta mżawka",
    56: "mżawka marznąca", 57: "gęsta mżawka marznąca",
    61: "słaby deszcz", 63: "deszcz", 65: "silny deszcz",
    66: "deszcz marznący", 67: "silny deszcz marznący",
    71: "słaby śnieg", 73: "śnieg", 75: "silny śnieg", 77: "śnieg ziarnisty",
    80: "przelotny deszcz", 81: "przelotny deszcz", 82: "silny przelotny deszcz",
    85: "przelotny śnieg", 86: "silny przelotny śnieg",
    95: "burza", 96: "burza z gradem", 99: "silna burza z gradem",
}
_WMO_EN: Final[dict[int, str]] = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "drizzle", 55: "dense drizzle",
    56: "freezing drizzle", 57: "dense freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm with hail",
}


class WeatherArgs(ToolArgs):
    location: str = Field(default="", max_length=120)


class ForecastArgs(ToolArgs):
    location: str = Field(default="", max_length=120)
    days: int = Field(default=3, ge=1, le=7)


def describe_code(code: int, *, language: str) -> str:
    table = _WMO_PL if language == "pl" else _WMO_EN
    return table.get(int(code), "brak opisu" if language == "pl" else "no description")


class _WeatherTool[ArgsT: ToolArgs](BaseTool[ArgsT]):
    """Baza: wspólne geokodowanie, jednostki i obsługa braku sieci."""

    def __init__(
        self,
        spec: ToolSpec,
        *,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(spec)
        self._settings = settings or get_settings()
        self._client = client

    def available(self) -> tuple[bool, str]:
        usable, reason = network_available(self._settings)
        if not usable:
            return usable, reason
        if self._settings.weather_provider == "none":
            return False, "pogoda jest wyłączona (WEATHER_PROVIDER=none)"
        return True, ""

    @property
    def metric(self) -> bool:
        return self._settings.weather_units != "imperial"

    def _units(self) -> dict[str, str]:
        if self.metric:
            return {"temperature_unit": "celsius", "wind_speed_unit": "kmh"}
        return {"temperature_unit": "fahrenheit", "wind_speed_unit": "mph"}

    def _labels(self) -> tuple[str, str]:
        return ("°C", "km/h") if self.metric else ("°F", "mph")

    async def _json(self, url: str, params: dict[str, Any]) -> Any:
        try:
            return await fetch_json(
                url,
                policy=HttpPolicy.from_settings(self._settings),
                params=params,
                client=self._client,
            )
        except NetworkError as exc:
            raise ToolError(exc.user_message) from exc

    async def _place(self, requested: str, language: str) -> dict[str, Any]:
        """Nazwa miejsca → współrzędne. Bez sieci nie ma jak tego zrobić."""
        name = (requested or self._settings.weather_default_location).strip()
        if not name:
            raise ToolError(
                "nie wiem, o jakie miejsce chodzi — podaj miasto albo ustaw "
                "WEATHER_DEFAULT_LOCATION w .env"
            )
        payload = await self._json(
            _GEOCODE_URL,
            {"name": name, "count": 1, "language": "pl" if language == "pl" else "en"},
        )
        results = payload.get("results") if isinstance(payload, dict) else None
        if not results:
            raise ToolError(f"nie znalazłam miejsca o nazwie '{name}'")
        first = results[0]
        return {
            "name": str(first.get("name") or name),
            "country": str(first.get("country") or ""),
            "latitude": float(first.get("latitude") or 0.0),
            "longitude": float(first.get("longitude") or 0.0),
            "timezone": str(first.get("timezone") or "auto"),
        }


class WeatherNowTool(_WeatherTool[WeatherArgs]):
    """``weather.current`` — pogoda teraz."""

    async def run(self, args: WeatherArgs, ctx: ToolContext) -> ToolResult:
        place = await self._place(args.location, ctx.language)
        payload = await self._json(
            _FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "precipitation,weather_code,wind_speed_10m",
                # „auto" = strefa miejsca, o które pytamy. Nie wysyłamy strefy
                # maszyny: pogoda w Tokio ma być podana w czasie Tokio.
                "timezone": "auto",
                **self._units(),
            },
        )
        current = payload.get("current") if isinstance(payload, dict) else None
        if not isinstance(current, dict):
            raise ToolError("serwis pogodowy nie zwrócił bieżących pomiarów")

        temperature_unit, wind_unit = self._labels()
        code = int(current.get("weather_code") or 0)
        description = describe_code(code, language=ctx.language)
        where = f"{place['name']}{', ' + place['country'] if place['country'] else ''}"
        data = {
            "location": where,
            "description": description,
            "temperature": current.get("temperature_2m"),
            "feels_like": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "wind": current.get("wind_speed_10m"),
            "units": {"temperature": temperature_unit, "wind": wind_unit},
            "observed_at": current.get("time"),
            "source": "open-meteo.com",
        }
        # Opis dla człowieka musi być w JĘZYKU ODPOWIEDZI — inaczej wychodzi
        # zlepek („light rain, odczuwalna 15°C, wiatr 20 km/h"), który asystent
        # potem czyta na głos. Zauważone przy uruchomieniu na żywo.
        if ctx.language == "pl":
            display = (
                f"{where}: {description}, {data['temperature']}{temperature_unit} "
                f"(odczuwalna {data['feels_like']}{temperature_unit}), wiatr "
                f"{data['wind']} {wind_unit}"
            )
        else:
            display = (
                f"{where}: {description}, {data['temperature']}{temperature_unit} "
                f"(feels like {data['feels_like']}{temperature_unit}), wind "
                f"{data['wind']} {wind_unit}"
            )
        return ToolResult.success(data, display=display, untrusted=True)


class WeatherForecastTool(_WeatherTool[ForecastArgs]):
    """``weather.forecast`` — prognoza na kilka dni."""

    async def run(self, args: ForecastArgs, ctx: ToolContext) -> ToolResult:
        place = await self._place(args.location, ctx.language)
        payload = await self._json(
            _FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_sum,wind_speed_10m_max",
                "forecast_days": args.days,
                "timezone": "auto",
                **self._units(),
            },
        )
        daily = payload.get("daily") if isinstance(payload, dict) else None
        if not isinstance(daily, dict) or not daily.get("time"):
            raise ToolError("serwis pogodowy nie zwrócił prognozy")

        temperature_unit, wind_unit = self._labels()
        days: list[dict[str, Any]] = []
        times = list(daily.get("time") or [])
        for index, day in enumerate(times[: args.days]):
            days.append(
                {
                    "date": day,
                    "description": describe_code(
                        int(_at(daily.get("weather_code"), index) or 0), language=ctx.language
                    ),
                    "max": _at(daily.get("temperature_2m_max"), index),
                    "min": _at(daily.get("temperature_2m_min"), index),
                    "precipitation": _at(daily.get("precipitation_sum"), index),
                    "wind_max": _at(daily.get("wind_speed_10m_max"), index),
                }
            )

        where = f"{place['name']}{', ' + place['country'] if place['country'] else ''}"
        summary = "; ".join(
            f"{item['date']}: {item['description']} {item['min']}–{item['max']}{temperature_unit}"
            for item in days
        )
        return ToolResult.success(
            {
                "location": where,
                "days": days,
                "units": {"temperature": temperature_unit, "wind": wind_unit},
                "source": "open-meteo.com",
            },
            display=f"{where} — {summary}",
            untrusted=True,
        )


def _at(values: Any, index: int) -> Any:
    if isinstance(values, list) and 0 <= index < len(values):
        return values[index]
    return None


def build_weather_tools(
    settings: Settings | None = None, *, client: httpx.AsyncClient | None = None
) -> Sequence[Tool[Any]]:
    """Narzędzia pogodowe (Open-Meteo, bez klucza API)."""
    active = settings or get_settings()
    default = active.weather_default_location.strip()
    hint = f" Default location when none is given: {default}." if default else ""
    return (
        WeatherNowTool(
            ToolSpec(
                name="weather.current",
                description=(
                    "Current weather for a place: temperature, how it feels, wind, "
                    "precipitation. Always call this instead of guessing — you cannot know "
                    "today's weather." + hint
                ),
                summary="pogoda teraz",
                args_model=WeatherArgs,
                risk=RiskLevel.MEDIUM,
                requires_network=True,
                timeout_s=min(60.0, active.web_timeout_s + 10.0),
            ),
            settings=active,
            client=client,
        ),
        WeatherForecastTool(
            ToolSpec(
                name="weather.forecast",
                description=(
                    "Weather forecast for the next few days (1-7) for a place: description, "
                    "minimum and maximum temperature, precipitation." + hint
                ),
                summary="prognoza pogody",
                args_model=ForecastArgs,
                risk=RiskLevel.MEDIUM,
                requires_network=True,
                timeout_s=min(60.0, active.web_timeout_s + 10.0),
            ),
            settings=active,
            client=client,
        ),
    )


__all__ = [
    "ForecastArgs",
    "WeatherArgs",
    "WeatherForecastTool",
    "WeatherNowTool",
    "build_weather_tools",
    "describe_code",
]
