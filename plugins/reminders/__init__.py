"""Plugin przypomnień i timerów — w całości lokalny (Faza 11).

„Przypomnij mi za 20 minut o praniu", „obudź mnie o 7". Nic z tego nie wymaga
internetu, żadnego konta i żadnej usługi w chmurze — a mimo to jest to jedna z
rzeczy, po które sięga się najczęściej. Ten plugin jest tu również jako dowód
tezy: **plugin nie musi niczego pobierać z sieci, żeby był użyteczny**.

Jak to działa
-------------

1. Model tłumaczy zdanie użytkownika na wywołanie ``reminders.add`` z konkretnym
   terminem (``in_minutes`` albo ``at``),
2. narzędzie zapisuje wpis do **bazy z Fazy 5** (osobna tabela ``plugin_reminders``),
   więc plan przeżywa zamknięcie programu,
3. interfejs co jakiś czas woła :meth:`RemindersPlugin.poll`, a ten oddaje
   przypomnienia, których termin minął — i od razu oznacza je jako zrealizowane,
   żeby nie odezwały się drugi raz.

Krok 3 jest tym, co odróżnia działające przypomnienie od zapisanej notatki.
Świadomie nie ma tu wątku ani ``asyncio.sleep`` do terminu: proces bywa zamykany,
a wtedy budzik w pamięci znika razem z nim. Zapytanie bazy o „co już minęło" jest
odporne na restart, przestawienie zegara i uśpienie komputera.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from i18n import t
from plugins.manager import BasePlugin, PluginContext, PluginInfo, PluginNotice
from plugins.reminders.storage import Reminder, ReminderError, ReminderStore
from plugins.reminders.tools import build_reminder_tools
from tools.base import Tool

logger = logging.getLogger(__name__)

INFO = PluginInfo(
    name="reminders",
    description="Przypomnienia i timery — lokalnie, bez internetu.",
    version="1.0",
    requires="pamięci trwałej (SQLite z Fazy 5)",
)


def notice_text(reminder: Reminder) -> str:
    """Zdanie, które zobaczy (albo usłyszy) użytkownik.

    Przez katalog tłumaczeń, a nie na sztywno: to jedyny tekst tego pluginu,
    który trafia wprost do człowieka — i bywa CZYTANY NA GŁOS, więc musi być w
    jego języku, a nie w języku autora pluginu.
    """
    return t("plugins.reminders.notice", text=reminder.text)


class RemindersPlugin(BasePlugin):
    """Przypomnienia zapisane w bazie asystenta."""

    def __init__(self) -> None:
        super().__init__(INFO)
        self._store: ReminderStore | None = None
        self._store_for: Any | None = None

    # --- kontrakt pluginu -------------------------------------------------- #

    def available(self, ctx: PluginContext) -> tuple[bool, str]:
        if ctx.database is None:
            return False, (
                "brak pamięci trwałej — przypomnienia muszą przetrwać restart, "
                "więc bez bazy plugin nie ma sensu (MEMORY_ENABLED=true)"
            )
        return True, ""

    def tools(self, ctx: PluginContext) -> Sequence[Tool[Any]]:
        store = self._get_store(ctx)
        return build_reminder_tools(store, max_active=ctx.settings.reminders_max_active)

    def poll(self, ctx: PluginContext) -> Sequence[PluginNotice]:
        """Przypomnienia, których termin właśnie minął.

        Oznaczenie „zrealizowane" dzieje się TUTAJ, a nie w interfejsie: gdyby
        zależało od tego, czy interfejs zdążył pokazać komunikat, ten sam budzik
        dzwoniłby w kółko przy każdym sprawdzeniu.
        """
        store = self._get_store(ctx)
        if store is None:
            return ()

        try:
            due = store.due(ctx.now())
        except ReminderError as exc:
            logger.warning("Nie mogę sprawdzić przypomnień: %s", exc)
            return ()
        except Exception as exc:  # baza zamknięta w trakcie zamykania programu
            logger.debug("Sprawdzenie przypomnień pominięte: %s", exc)
            return ()

        notices: list[PluginNotice] = []
        for reminder in due:
            store.mark_fired(reminder.id, now=ctx.now())
            notices.append(
                PluginNotice(
                    plugin=self.info.name,
                    text=notice_text(reminder),
                    kind="reminder",
                    # Przypomnienie ma sens tylko wtedy, gdy da się je zauważyć
                    # bez patrzenia w okno — dlatego domyślnie idzie na głos.
                    speak=True,
                    data={"id": reminder.id, "action": reminder.action},
                )
            )

        if notices:
            # Sprzątanie przy okazji: tabela nie ma rosnąć bez końca, a nie warto
            # budzić osobnego zadania po to, żeby raz na jakiś czas coś skasować.
            try:
                store.purge_older_than(ctx.settings.reminders_keep_days, now=ctx.now())
            except Exception as exc:  # pragma: no cover - sprzątanie jest opcjonalne
                logger.debug("Sprzątanie starych przypomnień pominięte: %s", exc)
        return notices

    # --- części składowe --------------------------------------------------- #

    def _get_store(self, ctx: PluginContext) -> ReminderStore | None:
        """Magazyn dla TEJ bazy. Zmiana bazy (np. w teście) buduje nowy."""
        if ctx.database is None:
            return None
        if self._store is not None and self._store_for is ctx.database:
            return self._store
        try:
            self._store = ReminderStore(ctx.database)
            self._store_for = ctx.database
        except ReminderError as exc:  # pragma: no cover - brak bazy łapie available()
            logger.warning("Przypomnienia niedostępne: %s", exc)
            return None
        return self._store


PLUGIN = RemindersPlugin()


def create_plugin() -> RemindersPlugin:
    return PLUGIN


__all__ = ["INFO", "PLUGIN", "RemindersPlugin", "create_plugin", "notice_text"]
