---
name: Propozycja zmiany / Feature request
about: Pomysł na nową funkcję albo na zmianę istniejącej
title: ''
labels: enhancement
assignees: ''
---

## Problem

<!-- Zacznij od problemu, nie od rozwiązania. Co próbujesz zrobić i co Ci
     w tym przeszkadza? Bez tego trudno ocenić, czy proponowane rozwiązanie
     jest najlepsze z możliwych. -->

## Propozycja

<!-- Jak miałoby to działać z punktu widzenia użytkownika. -->

## Czego to dotyczy

- [ ] nowe **narzędzie** (coś, co model może wywołać)
- [ ] nowy **plugin**
- [ ] rozpoznawanie mowy / słowo aktywujące
- [ ] synteza mowy
- [ ] pamięć (długoterminowa albo semantyczna)
- [ ] interfejs (okno, terminal, tryb usługi)
- [ ] instalacja i konfiguracja
- [ ] coś innego:

## Jeśli to nowe narzędzie

<!-- Wypełnij, jeśli chodzi o coś, co model ma móc wywołać. -->

**Poziom ryzyka** — SAFE (tylko odczyt) / MEDIUM (zmiana odwracalna) /
HIGH (skutków nie da się cofnąć) / CRITICAL (może zepsuć system):

**Czy wymaga czegoś, czego nie ma na czystej instalacji** (program systemowy,
klucz API, sprzęt)?

## Granice projektu

Sprawdź, czy pomysł nie trafia w coś, co jest tu **świadomą decyzją**, a nie
brakiem — pełna lista w sekcji Ograniczeń w README:

- [ ] Model **nigdy** nie wykonuje niczego poza zdefiniowanymi narzędziami.
      Nie będzie „napisz i uruchom skrypt" ani dowolnej powłoki.
- [ ] HIGH i CRITICAL **zawsze** wymagają potwierdzenia. Nie będzie trybu
      „ufam ci, nie pytaj".
- [ ] Architektura jest **jednoosobowa i jednomaszynowa**: bez kont, bez
      chmury, bez synchronizacji między urządzeniami, bez dostępu zdalnego.
- [ ] Wszystko liczy się **lokalnie**. Propozycje wysyłania treści rozmów
      albo pamięci do zewnętrznego API są poza zakresem projektu.

<!-- Jeśli Twój pomysł wymaga przesunięcia którejś z tych granic — napisz to
     wprost i uzasadnij. To nie znaczy „nie", znaczy „porozmawiajmy". -->
