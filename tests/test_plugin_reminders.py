"""Testy pluginu przypomnień (Faza 11).

Baza jest prawdziwa, ale leży w ``tmp_path`` — nic tu nie dotyka bazy asystenta
na maszynie dewelopera. Zegar jest wstrzykiwany, więc „za 20 minut" da się
sprawdzić bez czekania dwudziestu minut.

Sprawdzamy trzy rzeczy, o które chodzi w tym pluginie:

1. przypomnienie **przeżywa restart** (nowy obiekt magazynu na tym samym pliku),
2. **odzywa się raz** — po zadzwonieniu nie dzwoni w kółko przy każdym sprawdzeniu,
3. zaplanowanie AKCJI podnosi ryzyko z SAFE na MEDIUM.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from config import Settings
from database.database import Database
from plugins.manager import PluginContext
from plugins.reminders import RemindersPlugin
from plugins.reminders.storage import STATE_CANCELLED, STATE_FIRED, ReminderStore
from plugins.reminders.tools import (
    ReminderAddArgs,
    ReminderCancelArgs,
    ReminderListArgs,
    build_reminder_tools,
    resolve_due,
)
from security.risk import RiskLevel
from tools.base import ToolContext, ToolError

# Poniedziałek, 17 sierpnia 2026, 12:00 UTC — stały punkt odniesienia.
NOW = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def clock(moment: datetime = NOW) -> Any:
    return lambda: moment


@pytest.fixture
def database(tmp_path: Path) -> Any:
    db = Database(tmp_path / "asystent.sqlite3")
    yield db
    db.close()


@pytest.fixture
def store(database: Any) -> ReminderStore:
    return ReminderStore(database)


def context(database: Any, *, now: datetime = NOW, **overrides: Any) -> PluginContext:
    return PluginContext(settings=make_settings(**overrides), database=database, now=clock(now))


def tool_context(**overrides: Any) -> ToolContext:
    return ToolContext(settings=make_settings(), now=clock(), **overrides)


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


# --------------------------------------------------------------------------- #
# Zamiana „za 20 minut" i „o 7" na konkretny moment
# --------------------------------------------------------------------------- #


def test_termin_wzgledny() -> None:
    assert resolve_due(now=NOW, in_minutes=20) == NOW + timedelta(minutes=20)


def test_godzina_ktora_dopiero_bedzie_to_dzis() -> None:
    local = NOW.astimezone()
    later = (local + timedelta(hours=2)).strftime("%H:%M")

    due = resolve_due(now=NOW, at=later)

    assert due > NOW
    assert due.astimezone().strftime("%H:%M") == later


def test_godzina_ktora_juz_minela_to_jutro() -> None:
    """„Obudź mnie o 7" powiedziane po siódmej znaczy jutro — nie za rok i nie w przeszłości."""
    local = NOW.astimezone()
    earlier = (local - timedelta(hours=2)).strftime("%H:%M")

    due = resolve_due(now=NOW, at=earlier)

    assert NOW < due <= NOW + timedelta(days=1)
    assert due.astimezone().strftime("%H:%M") == earlier


def test_data_bez_strefy_jest_czasem_lokalnym() -> None:
    """Data od użytkownika to jego czas, nie UTC — inaczej budzik trafi obok."""
    due = resolve_due(now=NOW, at="2026-08-20 07:00")

    assert due.astimezone().strftime("%Y-%m-%d %H:%M") == "2026-08-20 07:00"


@pytest.mark.parametrize("value", ["25:00", "wieczorem", "7", "jutro rano"])
def test_niezrozumialy_termin_konczy_sie_bledem_dla_modelu(value: str) -> None:
    with pytest.raises(ToolError):
        resolve_due(now=NOW, at=value)


def test_dwa_terminy_naraz_to_blad() -> None:
    with pytest.raises(ToolError):
        resolve_due(now=NOW, in_minutes=5, at="07:00")


# --------------------------------------------------------------------------- #
# Magazyn
# --------------------------------------------------------------------------- #


def test_przypomnienie_przezywa_restart(tmp_path: Path) -> None:
    """Najważniejsza własność: budzik ma zadziałać także po zamknięciu programu."""
    path = tmp_path / "asystent.sqlite3"

    first = Database(path)
    ReminderStore(first).add("pranie", NOW + timedelta(minutes=20), now=NOW)
    first.close()

    # „Restart": nowe połączenie, nowy magazyn, ten sam plik.
    second = Database(path)
    try:
        active = ReminderStore(second).active()
    finally:
        second.close()

    assert [item.text for item in active] == ["pranie"]


def test_termin_ktory_minal_trafia_do_due(store: ReminderStore) -> None:
    store.add("pranie", NOW - timedelta(minutes=1), now=NOW)
    store.add("później", NOW + timedelta(hours=3), now=NOW)

    due = store.due(NOW)

    assert [item.text for item in due] == ["pranie"]


def test_zrealizowane_nie_wraca(store: ReminderStore) -> None:
    saved = store.add("pranie", NOW - timedelta(minutes=1), now=NOW)

    store.mark_fired(saved.id, now=NOW)

    assert store.due(NOW) == []
    assert store.get(saved.id).state == STATE_FIRED  # type: ignore[union-attr]


def test_odwolane_nie_dzwoni(store: ReminderStore) -> None:
    saved = store.add("pranie", NOW - timedelta(minutes=1), now=NOW)

    cancelled = store.cancel(saved.id)

    assert cancelled is not None
    assert store.due(NOW) == []
    assert store.get(saved.id).state == STATE_CANCELLED  # type: ignore[union-attr]


def test_odwolanie_nieistniejacego_nic_nie_psuje(store: ReminderStore) -> None:
    assert store.cancel(4242) is None


def test_sprzatanie_zostawia_aktywne(store: ReminderStore) -> None:
    stary = store.add("stare", NOW - timedelta(days=60), now=NOW - timedelta(days=60))
    store.mark_fired(stary.id, now=NOW - timedelta(days=60))
    store.add("przyszłe", NOW + timedelta(days=1), now=NOW)

    usuniete = store.purge_older_than(30, now=NOW)

    assert usuniete == 1
    assert [item.text for item in store.active()] == ["przyszłe"]


# --------------------------------------------------------------------------- #
# Narzędzia
# --------------------------------------------------------------------------- #


def tools_for(store: ReminderStore, **overrides: Any) -> dict[str, Any]:
    built = build_reminder_tools(store, **overrides)
    return {tool.spec.name: tool for tool in built}


def test_dodanie_przypomnienia_zwraca_termin(store: ReminderStore) -> None:
    tool = tools_for(store)["reminders.add"]

    result = run(tool.run(ReminderAddArgs(text="pranie", in_minutes=20), tool_context()))

    assert result.ok
    assert result.data["text"] == "pranie"
    assert result.data["due_at"] == (NOW + timedelta(minutes=20)).isoformat()
    assert store.active()[0].text == "pranie"


def test_samo_przypomnienie_jest_safe_a_z_akcja_medium(store: ReminderStore) -> None:
    """Poziom ryzyka zależy od tego, czy przypomnienie ma coś ZROBIĆ."""
    tool = tools_for(store)["reminders.add"]

    samo = tool.effective_risk(ReminderAddArgs(text="pranie", in_minutes=20))
    z_akcja = tool.effective_risk(
        ReminderAddArgs(text="zgaś światło", in_minutes=20, action="ha.switch")
    )

    assert samo is RiskLevel.SAFE
    assert z_akcja is RiskLevel.MEDIUM
    # …a pytanie o zgodę pojawia się tylko w tym drugim przypadku.
    assert tool.confirmation(ReminderAddArgs(text="pranie", in_minutes=20)) is None
    assert (
        tool.confirmation(ReminderAddArgs(text="zgaś światło", in_minutes=20, action="ha.switch"))
        is not None
    )


def test_limit_aktywnych_przypomnien(store: ReminderStore) -> None:
    """Model potrafi wpaść w pętlę planowania — limit jest na to, nie na użytkownika."""
    tool = tools_for(store, max_active=2)["reminders.add"]
    for numer in range(2):
        run(tool.run(ReminderAddArgs(text=f"raz {numer}", in_minutes=numer + 1), tool_context()))

    with pytest.raises(ToolError, match="already"):
        run(tool.run(ReminderAddArgs(text="jeszcze", in_minutes=5), tool_context()))


def test_termin_za_daleko_jest_odrzucany(store: ReminderStore) -> None:
    tool = tools_for(store)["reminders.add"]

    with pytest.raises(ToolError):
        run(tool.run(ReminderAddArgs(text="kiedyś", at="2099-01-01 07:00"), tool_context()))


def test_lista_pokazuje_zaplanowane(store: ReminderStore) -> None:
    store.add("pranie", NOW + timedelta(minutes=20), now=NOW)
    tool = tools_for(store)["reminders.list"]

    result = run(tool.run(ReminderListArgs(), tool_context()))

    assert result.ok and len(result.data["reminders"]) == 1
    assert "pranie" in result.display


def test_pusta_lista_mowi_wprost(store: ReminderStore) -> None:
    tool = tools_for(store)["reminders.list"]

    result = run(tool.run(ReminderListArgs(), tool_context()))

    assert result.ok and result.data["reminders"] == []
    assert "no reminders" in result.display


def test_odwolanie_nieistniejacego_numeru_to_blad_dla_modelu(store: ReminderStore) -> None:
    tool = tools_for(store)["reminders.cancel"]

    with pytest.raises(ToolError, match="no active reminder"):
        run(tool.run(ReminderCancelArgs(reminder_id=999), tool_context()))


def test_bez_bazy_narzedzia_sa_niedostepne() -> None:
    """Bez pamięci trwałej model nie ma nawet widzieć tych narzędzi."""
    for tool in build_reminder_tools(None):
        usable, reason = tool.available()
        assert not usable and "pamięci trwałej" in reason


# --------------------------------------------------------------------------- #
# Plugin jako całość
# --------------------------------------------------------------------------- #


def test_plugin_bez_bazy_mowi_dlaczego_nie_dziala() -> None:
    plugin = RemindersPlugin()

    usable, reason = plugin.available(PluginContext(settings=make_settings()))

    assert not usable and "restart" in reason


def test_poll_oddaje_przypomnienie_i_oznacza_je(database: Any) -> None:
    plugin = RemindersPlugin()
    ctx = context(database)
    store = ReminderStore(database)
    store.add("pranie", NOW - timedelta(minutes=1), now=NOW)

    first = plugin.poll(ctx)
    second = plugin.poll(ctx)

    # Tekst idzie przez katalog tłumaczeń — testy startują z angielskim UI.
    assert [notice.text for notice in first] == ["Reminder: pranie"]
    assert first[0].kind == "reminder" and first[0].speak
    # Drugie sprawdzenie ma być ciche — inaczej budzik dzwoniłby bez końca.
    assert list(second) == []


def test_poll_nie_rusza_przyszlych(database: Any) -> None:
    plugin = RemindersPlugin()
    ctx = context(database)
    ReminderStore(database).add("później", NOW + timedelta(hours=1), now=NOW)

    assert list(plugin.poll(ctx)) == []


def test_plugin_daje_narzedzia_gdy_baza_jest(database: Any) -> None:
    plugin = RemindersPlugin()
    ctx = context(database)

    names = {tool.spec.name for tool in plugin.tools(ctx)}

    assert names == {"reminders.add", "reminders.list", "reminders.cancel"}
    assert plugin.available(ctx)[0]


def test_tabela_pluginu_ma_wlasny_przedrostek(database: Any) -> None:
    """Tabele pluginów nie mogą mieszać się z tabelami asystenta."""
    ReminderStore(database).ensure_schema()

    rows = database.query("SELECT name FROM sqlite_master WHERE type = 'table'")
    names = {str(row["name"]) for row in rows}

    assert "plugin_reminders" in names
