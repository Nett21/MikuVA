"""Sprawdzenia środowiska audio dopisane do wspólnego mechanizmu z Fazy 1.

Import tego modułu rejestruje dodatkowe pozycje w ``detect_dependencies()``,
więc pojawiają się one automatycznie w ``python main.py --check-deps`` i w
``config/dependency_status.json``. Żadna z nich nie jest wymagana — brak
mikrofonu czy modelu ma wyłączyć tryb głosowy, a nie zatrzymać asystenta.
"""

from __future__ import annotations

import importlib.util
import logging

from config import (
    WAKEWORD_DIR,
    WHISPER_CACHE_DIR,
    DependencyCheck,
    DependencyContext,
    find_local_whisper_model,
    find_piper_binary,
    iter_local_whisper_models,
    pip_install_hint,
    piper_voice_directories,
    register_dependency_check,
    resolve_compute_device,
)
from i18n import t

logger = logging.getLogger(__name__)


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _check_microphone(context: DependencyContext) -> DependencyCheck:
    """Sprawdź realną dostępność urządzenia wejściowego (nie tylko pakietu)."""
    if not context.settings.mic_enabled:
        return DependencyCheck(
            name=t("deps.mic.name"),
            category="hardware",
            required=False,
            ok=False,
            detail=t("deps.mic.disabled"),
            hint=t("deps.mic.disabled_hint"),
            phase=2,
        )

    if not _module_available("sounddevice") or not _module_available("numpy"):
        return DependencyCheck(
            name=t("deps.mic.name"),
            category="hardware",
            required=False,
            ok=False,
            detail=t("deps.mic.no_packages"),
            hint=pip_install_hint(context.offline),
            phase=2,
        )

    # Import lokalny: gdyby PortAudio nie było zainstalowane w systemie, ten
    # import rzuca OSError już na poziomie modułu sounddevice.
    from audio.microphone import MicrophoneError, list_input_devices

    try:
        devices = list_input_devices(context.settings)
    except MicrophoneError as exc:
        return DependencyCheck(
            name=t("deps.mic.name"),
            category="hardware",
            required=False,
            ok=False,
            detail=exc.message,
            hint=exc.hint or t("deps.mic.error_hint"),
            phase=2,
        )

    if not devices:
        return DependencyCheck(
            name=t("deps.mic.name"),
            category="hardware",
            required=False,
            ok=False,
            detail=t("deps.mic.no_devices"),
            hint=t("deps.mic.no_devices_hint"),
            phase=2,
        )

    wanted = context.settings.audio_input_device
    selected = devices[0]
    if wanted:
        match = next(
            (device for device in devices if wanted.lower() in device.name.lower()), None
        )
        if match is None:
            return DependencyCheck(
                name=t("deps.mic.name"),
                category="hardware",
                required=False,
                ok=False,
                detail=t(
                    "deps.mic.no_match",
                    wanted=wanted,
                    devices=", ".join(device.name for device in devices[:5]),
                ),
                hint=t("deps.mic.no_match_hint"),
                phase=2,
            )
        selected = match

    return DependencyCheck(
        name=t("deps.mic.name"),
        category="hardware",
        required=False,
        ok=True,
        detail=t("deps.mic.ok", count=len(devices), selected=selected.describe()),
        path=selected.name,
        phase=2,
    )


def _check_vad(context: DependencyContext) -> DependencyCheck:
    webrtc = _module_available("webrtcvad")
    engine = context.settings.vad_engine

    if engine == "webrtc" and not webrtc:
        return DependencyCheck(
            name=t("deps.vad.name"),
            category="package",
            required=False,
            ok=False,
            detail=t("deps.vad.webrtc_missing"),
            hint=(
                "ustaw VAD_ENGINE=auto — wbudowany detektor energetyczny nie wymaga "
                "żadnych pakietów ani sieci"
                if context.offline
                else "pip install webrtcvad-wheels albo ustaw VAD_ENGINE=auto"
            ),
            phase=2,
        )

    if engine == "energy" or not webrtc:
        return DependencyCheck(
            name=t("deps.vad.name"),
            category="package",
            required=False,
            ok=True,
            detail=t("deps.vad.energy"),
            phase=2,
        )

    return DependencyCheck(
        name=t("deps.vad.name"),
        category="package",
        required=False,
        ok=True,
        detail=t("deps.vad.webrtc", level=context.settings.vad_aggressiveness),
        phase=2,
    )


def _missing_cuda_library() -> str:
    """Której biblioteki CUDA brakuje, żeby Whisper policzył na karcie.

    ``faster-whisper`` (ctranslate2) potrzebuje cuBLAS i cuDNN. Bez nich ładowanie
    na ``cuda`` kończy się wyjątkiem, kod schodzi na CPU — i do tej pory jedynym
    śladem był wpis w logu. Sprawdzamy to BEZ ładowania modelu: samo pytanie o
    bibliotekę jest tanie i nie ściąga niczego z sieci.
    """
    import ctypes.util

    for library in ("cublas", "cudnn"):
        if ctypes.util.find_library(library) is None:
            return f"lib{library}"
    return ""


def _check_whisper(context: DependencyContext) -> list[DependencyCheck]:
    checks: list[DependencyCheck] = []
    available = _module_available("faster_whisper")

    device = resolve_compute_device(context.settings.whisper_device, context.gpu)
    compute_type = context.settings.whisper_compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    checks.append(
        DependencyCheck(
            name=t("deps.whisper.name"),
            category="package",
            required=False,
            ok=available,
            detail=(
                t(
                    "deps.whisper.detail",
                    model=context.settings.whisper_model,
                    device=device,
                    compute=compute_type,
                )
                if available
                else t("deps.whisper.missing")
            ),
            hint="" if available else pip_install_hint(context.offline),
            phase=2,
        )
    )

    # Sam katalog cache nic nie znaczy — przerwane pobieranie zostawia w nim
    # puste migawki. Liczy się model, który faktycznie da się załadować.
    wanted = context.settings.whisper_model
    local = find_local_whisper_model(wanted, WHISPER_CACHE_DIR)
    others = sorted(iter_local_whisper_models(WHISPER_CACHE_DIR))

    if local is not None:
        detail = t("deps.whisper.local", model=wanted)
        hint = ""
    else:
        detail = t("deps.whisper.absent", model=wanted)
        if others:
            detail += t("deps.whisper.others", models=", ".join(others))
        hint = (
            t("deps.whisper.hint_offline")
            if context.offline
            else t("deps.whisper.hint_online")
        )

    # Karta jest, sterownik jest, a i tak liczy procesor — najczęstsza cicha
    # przyczyna wolnego rozpoznawania mowy.
    if device == "cuda":
        missing = _missing_cuda_library()
        if missing:
            from config import PackageManager

            checks.append(
                DependencyCheck(
                    name=t("deps.whisper.name"),
                    category="hardware",
                    required=False,
                    ok=False,
                    detail=t("deps.whisper.cuda_missing", library=missing),
                    hint=(
                        t("deps.whisper.cuda_hint_pacman")
                        if context.platform_info.package_manager is PackageManager.PACMAN
                        else t("deps.whisper.cuda_hint_generic")
                    ),
                    phase=2,
                )
            )

    checks.append(
        DependencyCheck(
            name=t("deps.whisper.cache_name"),
            category="model",
            required=False,
            ok=local is not None,
            detail=detail,
            path=str(local) if local is not None else str(WHISPER_CACHE_DIR),
            hint=hint,
            phase=2,
        )
    )
    return checks


def _check_wakeword(context: DependencyContext) -> list[DependencyCheck]:
    """Faza 3: fraza, silnik i model słowa aktywującego."""
    from audio.wakeword import find_wakeword_models  # noqa: PLC0415 - unika cyklu importów

    settings = context.settings
    user = context.user_settings
    phrase = user.effective_wake_word
    source = (
        t("deps.wake.source_file") if user.wake_word.strip() else t("deps.wake.source_name")
    )

    if not settings.wake_enabled or settings.wake_engine == "none":
        return [
            DependencyCheck(
                name=t("deps.wake.name"),
                category="feature",
                required=False,
                ok=False,
                detail=t("deps.wake.disabled", phrase=phrase, source=source),
                hint=t("deps.wake.disabled_hint"),
                phase=3,
            )
        ]

    models = find_wakeword_models(user)
    openwakeword_ready = _module_available("openwakeword") and bool(models)

    if openwakeword_ready:
        engine_detail = t(
            "deps.wake.openwakeword", models=", ".join(path.name for path in models)
        )
    else:
        wake_model = settings.wake_whisper_model.strip() or settings.whisper_model
        engine_detail = t("deps.wake.whisper_detector", model=wake_model)
        if settings.wake_engine == "openwakeword":
            return [
                DependencyCheck(
                    name=t("deps.wake.name"),
                    category="feature",
                    required=False,
                    ok=False,
                    detail=t("deps.wake.missing_openwakeword", path=WAKEWORD_DIR),
                    path=str(WAKEWORD_DIR),
                    hint=t("deps.wake.missing_hint"),
                    phase=3,
                )
            ]

    checks = [
        DependencyCheck(
            name=t("deps.wake.name"),
            category="feature",
            required=False,
            ok=True,
            detail=t("deps.wake.ok", phrase=phrase, source=source, engine=engine_detail),
            hint="",
            phase=3,
        )
    ]

    # Detektor whisperowy potrzebuje swojego modelu na dysku — bez sieci to
    # jedyny moment, w którym da się to zauważyć przed pierwszym zawołaniem.
    if not openwakeword_ready:
        wake_model = settings.wake_whisper_model.strip() or settings.whisper_model
        local = find_local_whisper_model(wake_model, WHISPER_CACHE_DIR)
        checks.append(
            DependencyCheck(
                name=t("deps.wake.model_name"),
                category="model",
                required=False,
                ok=local is not None,
                detail=(
                    t("deps.wake.model_present", model=wake_model)
                    if local is not None
                    else t("deps.wake.model_absent", model=wake_model)
                ),
                path=str(local) if local is not None else str(WHISPER_CACHE_DIR),
                hint=(
                    ""
                    if local is not None
                    else f"python scripts/prepare_offline.py --whisper-model {wake_model}"
                ),
                phase=3,
            )
        )
    return checks


def _check_speaker(context: DependencyContext) -> DependencyCheck:
    """Czy jest na czym odtworzyć mowę? Brak głośnika to normalny stan, nie błąd."""
    if not _module_available("sounddevice") or not _module_available("numpy"):
        return DependencyCheck(
            name=t("deps.speaker.name"),
            category="hardware",
            required=False,
            ok=False,
            detail=t("deps.speaker.no_packages"),
            hint=pip_install_hint(context.offline),
            phase=4,
        )

    from audio.output import AudioOutputError, find_output_device, list_output_devices

    try:
        devices = list_output_devices(context.settings)
    except AudioOutputError as exc:
        return DependencyCheck(
            name=t("deps.speaker.name"),
            category="hardware",
            required=False,
            ok=False,
            detail=exc.message,
            hint=exc.hint or t("deps.speaker.error_hint"),
            phase=4,
        )

    if not devices:
        return DependencyCheck(
            name=t("deps.speaker.name"),
            category="hardware",
            required=False,
            ok=False,
            detail=t("deps.speaker.no_devices"),
            hint=t("deps.speaker.no_devices_hint"),
            phase=4,
        )

    wanted = context.settings.audio_output_device
    selected = devices[0]
    if wanted:
        match = find_output_device(wanted, context.settings)
        if match is None:
            return DependencyCheck(
                name=t("deps.speaker.name"),
                category="hardware",
                required=False,
                ok=False,
                detail=(
                    t(
                        "deps.device_no_match",
                        name=wanted,
                        devices=", ".join(device.name for device in devices[:5]),
                    )
                ),
                hint=t("deps.speaker.no_match_hint"),
                phase=4,
            )
        selected = match

    return DependencyCheck(
        name=t("deps.speaker.name"),
        category="hardware",
        required=False,
        ok=True,
        detail=t("deps.speaker.ok", count=len(devices), selected=selected.describe()),
        path=selected.name,
        phase=4,
    )


def _check_tts(context: DependencyContext) -> list[DependencyCheck]:
    """Faza 4: silnik mowy i głos, którym asystent będzie mówił."""
    from audio.tts import iter_piper_voices, select_piper_voice  # noqa: PLC0415 - unika cyklu

    settings = context.settings
    user = context.user_settings

    if not settings.tts_enabled or not user.speaks:
        reason = (
            t("deps.tts.disabled_env")
            if not settings.tts_enabled
            else t("deps.tts.disabled_engine", engine=user.voice_engine)
        )
        return [
            DependencyCheck(
                name=t("deps.tts.name"),
                category="feature",
                required=False,
                ok=False,
                detail=reason,
                hint=t("deps.tts.disabled_hint"),
                phase=4,
            )
        ]

    package = _module_available("piper")
    binary = find_piper_binary(settings)
    if package:
        engine_detail = t("deps.tts.package")
        engine_path: str | None = None
    elif binary is not None:
        engine_detail = t("deps.tts.binary")
        engine_path = str(binary)
    else:
        engine_detail = t("deps.tts.missing")
        engine_path = None

    checks = [
        DependencyCheck(
            name=t("deps.tts.engine_name"),
            category="package",
            required=False,
            ok=package or binary is not None,
            detail=engine_detail,
            path=engine_path,
            hint=(
                ""
                if (package or binary is not None)
                else t("deps.tts.missing_hint")
            ),
            phase=4,
        )
    ]

    voices = iter_piper_voices(settings)
    directories = piper_voice_directories(settings)
    if voices:
        selected = select_piper_voice(settings, user, voices=voices)
        wanted = user.piper_model.strip()
        detail = t(
            "deps.voice.found",
            count=len(voices),
            voices=", ".join(voice.name for voice in voices[:5]),
        )
        hint = ""
        ok = True
        if wanted and (selected is None or selected.name.lower() != wanted.lower()):
            ok = False
            detail = t(
                "deps.voice.not_found",
                wanted=wanted,
                voices=", ".join(voice.name for voice in voices[:5]),
            )
            hint = t("deps.voice.not_found_hint")
        elif selected is not None:
            detail += t("deps.voice.selected", selected=selected.describe())
        checks.append(
            DependencyCheck(
                name=t("deps.voice.name"),
                category="model",
                required=False,
                ok=ok,
                detail=detail,
                path=str(selected.path) if selected is not None else str(directories[0]),
                hint=hint,
                phase=4,
            )
        )
    else:
        checks.append(
            DependencyCheck(
                name=t("deps.voice.name"),
                category="model",
                required=False,
                ok=False,
                detail=t(
                    "deps.voice.no_files",
                    directories=", ".join(str(item) for item in directories[:3]),
                ),
                path=str(directories[0]),
                hint=t("deps.voice.no_files_hint"),
                phase=4,
            )
        )

    checks.append(_check_speaker(context))
    checks.extend(_check_rvc(context))
    return checks


def _check_rvc(context: DependencyContext) -> list[DependencyCheck]:
    """Faza 15: model RVC i implementacja, która go policzy.

    Pokazujemy to TYLKO wtedy, gdy użytkownik o RVC poprosił. Dla kogoś, kto
    używa samego Pipera, byłyby to dwie pozycje na czerwono opisujące funkcję,
    której nie włączał — a raport ma prowadzić do naprawy, nie straszyć.
    """
    from audio.rvc import (  # noqa: PLC0415 - unika cyklu
        INSTALL_APPLIO_SCRIPT,
        available_rvc_backends,
        resolve_rvc_device,
    )

    user = context.user_settings
    rvc = user.rvc
    if user.voice_engine != "rvc_miku" and not rvc.enabled:
        return []

    checks: list[DependencyCheck] = []

    model = rvc.resolved_model_path
    missing = rvc.missing_files()
    if not rvc.enabled:
        detail, hint, ok, path = t("deps.rvc.disabled"), t("deps.rvc.disabled_hint"), False, None
    elif model is None:
        detail, hint, ok, path = t("deps.rvc.no_path"), t("deps.rvc.no_path_hint"), False, None
    elif missing:
        detail = t("deps.rvc.missing", paths=", ".join(str(item) for item in missing))
        hint, ok, path = t("deps.rvc.missing_hint"), False, str(model)
    else:
        index = rvc.resolved_index_path
        detail = t(
            "deps.rvc.found",
            index=str(index) if index is not None else t("deps.rvc.no_index"),
        )
        hint, ok, path = "", True, str(model)

    checks.append(
        DependencyCheck(
            name=t("deps.rvc.model_name"),
            category="model",
            required=False,
            ok=ok,
            detail=detail,
            path=path,
            hint=hint,
            phase=15,
        )
    )

    backends = available_rvc_backends(context.settings)
    wanted = context.settings.rvc_backend.strip()
    device = resolve_rvc_device(context.settings, context.gpu)
    if wanted:
        backend_ok, backend_detail = True, t("deps.rvc.backend_forced", backend=wanted)
    elif backends:
        backend_ok, backend_detail = True, t("deps.rvc.backend_found", backend=", ".join(backends))
    else:
        backend_ok, backend_detail = False, t("deps.rvc.backend_missing")

    checks.append(
        DependencyCheck(
            name=t("deps.rvc.backend_name"),
            category="package",
            required=False,
            ok=backend_ok,
            detail=f"{backend_detail} — {device.describe()}",
            hint=(
                t("deps.rvc.backend_missing_hint", script=INSTALL_APPLIO_SCRIPT)
                if not backend_ok
                else (t("deps.rvc.cpu_hint") if device.is_cpu else "")
            ),
            phase=15,
        )
    )
    return checks


@register_dependency_check
def check_audio_stack(context: DependencyContext) -> list[DependencyCheck]:
    """Sprawdzenia Faz 2, 3 i 4 zebrane w jedną pozycję rejestru."""
    checks: list[DependencyCheck] = [_check_microphone(context), _check_vad(context)]
    checks.extend(_check_whisper(context))
    checks.extend(_check_wakeword(context))
    checks.extend(_check_tts(context))
    return checks


__all__ = ["check_audio_stack"]
