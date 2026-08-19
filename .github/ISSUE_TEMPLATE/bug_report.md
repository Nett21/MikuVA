---
name: Zgłoszenie błędu / Bug report
about: Coś nie działa tak, jak opisuje README
title: ''
labels: bug
assignees: ''
---

## Co się dzieje

<!-- Jedno-dwa zdania. Co zrobiłeś i co się stało zamiast tego, czego oczekiwałeś. -->

## Jak to powtórzyć

1.
2.
3.

**Czego oczekiwałem:**
**Co się stało:**

## Środowisko

<!-- WKLEJ WYNIK TEJ KOMENDY. To jedna linijka, która opisuje całe środowisko
     (system, Python, pakiety, GPU, mikrofon, modele, Ollama) i oszczędza nam
     obu rundę pytań w tę i z powrotem. -->

```
$ python main.py --check-deps


```

## Logi

<!-- Pełne ślady wyjątków lądują w logs/errors.log, a nie na ekranie.
     Wklej ostatnie kilkanaście linii z okolic błędu.
     W trybie usługi: journalctl --user -u miku-assistant -n 50 -->

```


```

## Zanim wyślesz — trzy rzeczy

- [ ] Przeczytałem sekcję **Ograniczenia / Known limitations** w README.
      Halucynacje małego modelu, pomyłki rozpoznawania mowy w hałasie,
      pytanie o zgodę przy każdej akcji HIGH i brak działającego RVC to
      **udokumentowane właściwości**, nie błędy.
- [ ] Sprawdziłem, czy w tym, co wklejam, nie ma kluczy API, tokenów ani
      ścieżek z nazwą mojego konta.
- [ ] Problem występuje też po `python main.py --check-deps` bez błędów
      (albo dołączam raport pokazujący, czego brakuje).
