#!/usr/bin/env python3
"""Proces-pracownik RVC: trzyma model w pamięci i konwertuje pliki na żądanie.

Ten plik jest URUCHAMIANY PRZEZ INNY INTERPRETER niż reszta projektu i dlatego
nie importuje z niego ani jednej rzeczy. Powód jest przyziemny: biblioteki RVC
(``fairseq``, ``omegaconf``) nie działają na Pythonie nowszym niż 3.10, a
asystent działa na 3.12+. Zamiast cofać cały projekt o cztery wersje w dół,
trzymamy RVC w osobnym środowisku i rozmawiamy z nim przez potok.

Dlaczego proces trwały, a nie wywołanie na fragment: załadowanie modelu RVC to
kilka sekund i kilkaset megabajtów. Przy zdaniu co pół sekundy start procesu
za każdym razem kosztowałby więcej niż sama konwersja.

Protokół — po jednym obiekcie JSON w linii, w obie strony::

    ← {"cmd": "convert", "in": "a.wav", "out": "b.wav", "pitch": 12, "index_rate": 0.75}
    → {"ok": true}
    → {"ok": false, "error": "opis"}
    ← {"cmd": "quit"}

Zaraz po załadowaniu modelu leci linia powitalna::

    → {"ready": true, "device": "cuda:0"}
    → {"ready": false, "error": "opis"}

Na standardowe wyjście NIE WOLNO wypisać niczego poza protokołem, a torch,
fairseq i loguru mają zwyczaj pisać, gdzie popadnie. Dlatego prawdziwe
``stdout`` jest odkładane na bok, zanim cokolwiek zostanie zaimportowane,
a ``sys.stdout`` podmieniane na ``stderr``. Bez tego pierwszy komunikat
biblioteki rozjeżdżałby protokół.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from typing import Any, TextIO

#: Prawdziwe wyjście, odłożone na bok w :func:`przejmij_stdout`.
_PROTOCOL: TextIO | None = None


def przejmij_stdout() -> None:
    """Odłóż prawdziwe ``stdout`` na bok i podstaw pod nie ``stderr``.

    Wywoływane na początku :func:`main`, a nie przy imporcie: podmiana
    ``sys.stdout`` jako efekt uboczny importu psułaby wszystko, co ten plik
    zaimportuje — łącznie z pytestem. Wystarczy, że zdarzy się przed importem
    bibliotek RVC, a te wchodzą dopiero niżej.
    """
    global _PROTOCOL  # noqa: PLW0603 - stan procesu, jeden na cały program
    if _PROTOCOL is None:
        _PROTOCOL = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr


def zaufaj_lokalnym_checkpointom() -> None:
    """Pozwól ``fairseq`` wczytać HuBERT-a na PyTorchu 2.6+.

    Od PyTorcha 2.6 ``torch.load`` domyślnie czyta z ``weights_only=True``:
    z pliku wolno odtworzyć tensory, ale nie dowolne obiekty Pythona.
    Checkpoint HuBERT-a, którego RVC używa do wyciągania cech mowy, pochodzi
    z ``fairseq`` i niesie w środku ``fairseq.data.dictionary.Dictionary``
    razem z całą konfiguracją treningu. ``fairseq`` 0.12.2 woła ``torch.load``
    bez tego argumentu, więc na nowym PyTorchu kończy się to ``UnpicklingError``.

    Gorsze od samego błędu jest to, jak się objawia: ``rvc_python`` łapie go,
    zapisuje jako ostrzeżenie i zwraca krotkę zamiast dźwięku, a konwersja
    wywraca się dopiero linijkę dalej na ``'tuple' object has no attribute
    'dtype'``. Asystent po cichu wraca do zwykłego Pipera i nic nie wskazuje,
    że powodem jest sposób czytania jednego pliku.

    Dopisywanie kolejnych klas przez ``add_safe_globals`` nie kończy się na
    ``Dictionary`` — dalej są ``argparse.Namespace`` i typy z ``omegaconf``,
    a lista zmienia się z wersją checkpointu. Dlatego cofamy domyślne
    zachowanie tylko **w tym procesie**: to pracownik RVC, ładuje wyłącznie
    modele wskazane przez użytkownika w jego własnych ustawieniach i nie
    przyjmuje plików z sieci. Reszta asystenta pracuje w drugim interpreterze
    i tej zmiany nie widzi.

    Jawne ``weights_only`` w wywołaniu zawsze wygrywa — podmieniamy tylko
    wartość domyślną.
    """
    import torch  # noqa: PLC0415 - dopiero tutaj, po przejęciu stdout

    original = torch.load
    if getattr(original, "_mikuva_weights_only", False):
        return

    # Sygnatura celowo nijaka: podstawiamy się pod `torch.load`, którego lista
    # argumentów zmienia się z wersją na wersję, a naszą jedyną sprawą jest
    # jeden argument nazwany. Resztę przekazujemy dalej bez oglądania.
    def load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        try:
            return original(*args, **kwargs)
        except TypeError:
            # PyTorch starszy niż 1.13 nie zna tego argumentu — i nie musi,
            # bo tam domyślne zachowanie jest już takie, jakiego chcemy.
            kwargs.pop("weights_only", None)
            return original(*args, **kwargs)

    load._mikuva_weights_only = True  # type: ignore[attr-defined]
    torch.load = load  # type: ignore[assignment]


def send(payload: dict[str, object]) -> None:
    stream = _PROTOCOL if _PROTOCOL is not None else sys.__stdout__
    if stream is None:  # pragma: no cover - niemożliwe po przejmij_stdout()
        return
    stream.write(json.dumps(payload) + "\n")
    stream.flush()


# --------------------------------------------------------------------------- #
# Silniki
# --------------------------------------------------------------------------- #
#
# Dwa światy, jeden protokół. `rvc_python` stoi na `fairseq` i dlatego siedzi
# na Pythonie 3.10; Applio fairseq nie ma, za to szuka swoich wag względem
# katalogu roboczego. Pętla protokołu nie ma prawa o tym wiedzieć — woła
# `konwertuj` i tyle.


class SilnikRvcPython:
    """Pakiet ``rvc-python``: API obiektowe, model trzymany w pamięci."""

    def __init__(self, model: str, index: str, device: str, version: str) -> None:
        zaufaj_lokalnym_checkpointom()
        from rvc_python.infer import RVCInference  # noqa: PLC0415

        self._engine = RVCInference(device=device)
        self._engine.load_model(model, version=version, index_path=index or "")
        self._ostatnie: tuple[int, float] | None = None
        self.device = device

    def konwertuj(self, wejscie: str, wyjscie: str, pitch: int, index_rate: float) -> None:
        # Parametry ustawiamy tylko przy zmianie: w niektórych wersjach
        # `set_params` przelicza indeks, co przy fragmencie na pół sekundy
        # kosztuje więcej niż sama konwersja.
        if self._ostatnie != (pitch, index_rate):
            self._engine.set_params(f0up_key=pitch, index_rate=index_rate)
            self._ostatnie = (pitch, index_rate)
        self._engine.infer_file(wejscie, wyjscie)

    def zamknij(self) -> None:
        self._engine.unload_model()


class SilnikApplio:
    """Applio — ten sam RVC, ale bez ``fairseq`` i z własnym drzewem wag.

    Trzy rzeczy, które trzeba tu zrobić inaczej niż wszędzie:

    * **Katalog roboczy.** ``rvc/infer/infer.py`` wykonuje przy imporcie
      ``now_dir = os.getcwd()`` i po tym katalogu szuka embeddera oraz
      predyktora. Proces MUSI więc być uruchomiony z katalogu Applio —
      pilnuje tego wołający, a my to tylko sprawdzamy, bo pomyłka objawia
      się dopiero przy pierwszej konwersji, komunikatem o braku wag.
    * **Wybór karty.** ``Config`` Applio bierze ``cuda:0``, jeśli tylko torch
      widzi kartę, i nie pyta nikogo o zdanie. Zamiast podmieniać mu pola po
      inicjalizacji — co zostawiłoby przeliczone pod inną kartę bufory —
      zawężamy widok przez ``CUDA_VISIBLE_DEVICES``, zanim wejdzie torch.
    * **Model ładujemy z góry.** ``convert_audio`` zrobiłoby to samo przy
      pierwszym wywołaniu, ale wtedy koszt wczytania modelu doliczyłby się do
      pierwszego zdania asystenta, a linia ``ready`` skłamałaby o gotowości.
    """

    def __init__(
        self,
        model: str,
        index: str,
        device: str,
        f0_method: str,
        embedder: str,
    ) -> None:
        from rvc.infer.infer import VoiceConverter  # noqa: PLC0415

        self._engine = VoiceConverter()
        self._model = model
        self._index = index
        self._f0_method = f0_method
        self._embedder = embedder
        self.device = str(getattr(self._engine.config, "device", device))
        # Model i embedder do pamięci, jeszcze przed zgłoszeniem gotowości.
        self._engine.get_vc(model, 0)
        self._engine.load_hubert(embedder, None)
        self._engine.last_embedder_model = embedder

    def konwertuj(self, wejscie: str, wyjscie: str, pitch: int, index_rate: float) -> None:
        # `split_audio`, `clean_audio` i `post_process` są wyłączone celowo:
        # to obróbka pod całe nagrania, a my dostajemy pojedyncze zdania i
        # każdy z tych kroków dokłada opóźnienie do czegoś, co ma iść na żywo.
        self._engine.convert_audio(
            audio_input_path=wejscie,
            audio_output_path=wyjscie,
            model_path=self._model,
            index_path=self._index,
            pitch=pitch,
            index_rate=index_rate,
            f0_method=self._f0_method,
            embedder_model=self._embedder,
            export_format="WAV",
            split_audio=False,
            clean_audio=False,
            post_process=False,
            f0_autotune=False,
            sid=0,
        )

    def zamknij(self) -> None:
        pass


def ustaw_widoczna_karte(device: str) -> None:
    """Przetłumacz ``--device`` na ``CUDA_VISIBLE_DEVICES``.

    Applio nie przyjmuje urządzenia w argumencie — czyta je z torcha. Jedyny
    sposób, żeby uszanować wybór użytkownika, to zawęzić torchowi widok, i to
    ZANIM torch zostanie zaimportowany.
    """
    numer = device.rsplit(":", maxsplit=1)[-1] if ":" in device else ""
    if device.startswith("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device.startswith("cuda") and numer.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = numer


def main(argv: list[str] | None = None) -> int:
    przejmij_stdout()
    parser = argparse.ArgumentParser(description="Proces-pracownik konwersji głosu RVC")
    parser.add_argument("--model", required=True, help="plik .pth")
    parser.add_argument("--index", default="", help="plik .index (opcjonalny)")
    parser.add_argument("--device", default="cpu:0", help="cpu, cuda:0, ...")
    parser.add_argument("--version", default="v2", help="wersja architektury modelu")
    parser.add_argument(
        "--engine",
        default="rvc_python",
        choices=("rvc_python", "applio"),
        help="implementacja RVC do użycia",
    )
    parser.add_argument(
        "--applio-root", default="", help="katalog z kodem Applio (wymagany dla --engine applio)"
    )
    parser.add_argument("--f0-method", default="rmvpe", help="sposób wykrywania wysokości tonu")
    parser.add_argument("--embedder", default="contentvec", help="model cech mowy (Applio)")
    args = parser.parse_args(argv)

    # `rvc_python` oczekuje urządzenia w postaci „cpu:0"/„cuda:0"; samo „cpu"
    # w części wersji kończy się wyjątkiem przy parsowaniu.
    device = args.device if ":" in args.device else f"{args.device}:0"

    try:
        # Import po sparsowaniu argumentów, żeby `--help` działało także tam,
        # gdzie pakietu nie ma — i żeby jego brak był komunikatem protokołu,
        # a nie wyjątkiem przy starcie.
        silnik: SilnikRvcPython | SilnikApplio
        if args.engine == "applio":
            korzen = os.path.abspath(args.applio_root or os.getcwd())
            if not os.path.isdir(os.path.join(korzen, "rvc")):
                send({"ready": False, "error": f"to nie jest katalog Applio: {korzen}"})
                return 1
            # Applio dopisuje sobie do `sys.path` własny katalog roboczy, więc
            # ustawienie kartoteki musi wyprzedzić import — nie odwrotnie.
            os.chdir(korzen)
            if korzen not in sys.path:
                sys.path.insert(0, korzen)
            ustaw_widoczna_karte(device)
            silnik = SilnikApplio(args.model, args.index, device, args.f0_method, args.embedder)
        else:
            silnik = SilnikRvcPython(args.model, args.index, device, args.version)
    except Exception as exc:  # brak pakietu, zepsuta instalacja, brak wag
        send({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1

    send({"ready": True, "device": silnik.device, "engine": args.engine})

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            send({"ok": False, "error": f"niepoprawny JSON: {exc}"})
            continue

        if request.get("cmd") == "quit":
            break
        if request.get("cmd") != "convert":
            send({"ok": False, "error": f"nieznane polecenie: {request.get('cmd')!r}"})
            continue

        try:
            silnik.konwertuj(
                request["in"],
                request["out"],
                int(request.get("pitch", 0)),
                float(request.get("index_rate", 0.75)),
            )
        except Exception as exc:
            send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            continue
        send({"ok": True})

    # Sprzątanie nie ma prawa zmienić kodu wyjścia.
    with contextlib.suppress(Exception):
        silnik.zamknij()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
