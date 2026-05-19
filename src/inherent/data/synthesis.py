"""TTS-synthesize training data for intent heads with no public spoken data.

Heads requiring synthesis:
  - hasPhotoQuery
  - hasCreateDocIntent
  - hasPersonContext
  - hasEventContext
  - hasDeepResearchIntent
  - hasInsightIntent
  - hasBrowsingAgentIntent
  - hasCallingAgentIntent

Pipeline:
  1. LLM authors ~10k diverse prompts per head (templated + free-form).
  2. TTS engine renders each prompt with multiple voices/accents/speeds.
  3. Output is mixed into the training set with the head label set true.

License-clean TTS engines (do not use XTTS-v2 or non-Apache F5-TTS):
  - OpenVoice V2 (MIT) — primary, voice cloning
  - SparkAudio/Spark-TTS-0.5B
  - mrfakename/OpenF5-TTS-Base (Apache fork of F5-TTS) — current default (`openf5-tts`)
  - ModelScope CosyVoice2 (Apache-style) — for accent diversity

Candidates under benchmark (plug in via `synthetic_tts_engine` when integrated):
  - Supertone/supertonic-3 — https://huggingface.co/Supertone/supertonic-3
  - ResembleAI/Dramabox — https://huggingface.co/ResembleAI/Dramabox
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple

from .. import INTENT_HEAD_ORDER
from ..features.frontend import SAMPLE_RATE
from .mic_augment import augment_wav_file, mic_augment_enabled, parse_snr_db_range
from .tts_engines import OPENF5_TTS_ENGINE, SUPERTONIC_TTS_ENGINE, SUPPORTED_TTS_ENGINES


@dataclass(frozen=True)
class SyntheticSample:
    audio_path: Path
    transcript: str
    head: str
    voice_id: str
    tts_engine: str


SYNTHETIC_HEADS = (
    "hasPhotoQuery",
    "hasCreateDocIntent",
    "hasPersonContext",
    "hasEventContext",
    "hasDeepResearchIntent",
    "hasInsightIntent",
    "hasBrowsingAgentIntent",
    "hasCallingAgentIntent",
)

DEFAULT_VOICES = ("openvoice", "cosyvoice2")
DEFAULT_TTS_VOICE_DIR = Path("data/tts_voices")
OPENF5_TTS_COMMAND = "f5-tts_infer-cli"
OPENF5_MODEL_ENV = "INHERENT_OPENF5_MODEL"
APPROVED_OPENF5_MODEL_IDS = ("mrfakename/OpenF5-TTS-Base",)
DISALLOWED_TTS_MODEL_IDS = (
    "F5-TTS",
    "F5TTS_v1_Base",
    "SWivid/F5-TTS",
    "XTTS-v2",
    "coqui/XTTS-v2",
)


class OpenF5ModelFiles(NamedTuple):
    model_cfg: Path
    ckpt_file: Path
    vocab_file: Path

PROMPT_TEMPLATES_BY_HEAD: dict[str, list[str]] = {
    "hasPhotoQuery": [
        "show me photos of {subject}",
        "find pictures from {time_window}",
        "pull up that photo of {subject}",
        "search my gallery for {subject}",
        "bring up images from {time_window}",
        "look for screenshots about {topic}",
        "find the picture with {subject}",
        "show photos from {place}",
        "open my camera roll for {time_window}",
        "find images of {topic}",
    ],
    "hasCreateDocIntent": [
        "draft a document about {topic}",
        "write up a note on {topic}",
        "start a new doc for {topic}",
        "create a report about {topic}",
        "make a working document for {project}",
        "start writing the {document_type}",
        "turn this into a document about {topic}",
        "prepare a brief on {topic}",
        "open a blank doc for {project}",
        "create meeting notes for {event}",
    ],
    "hasPersonContext": [
        "what did {contact} say about {topic}",
        "find my notes about {contact}",
        "summarize my last conversation with {contact}",
        "what do i know about {contact}",
        "pull together context on {contact}",
        "when did i last talk to {contact}",
        "find messages from {contact} about {topic}",
        "what has {contact} asked me to do",
        "show me the important context for {contact}",
        "what should i remember about {contact}",
        "look up my history with {contact}",
        "find references to {contact} in my notes",
    ],
    "hasEventContext": [
        "what happened in {event}",
        "summarize the {event}",
        "what did we decide in {event}",
        "find notes from {event}",
        "what are the follow ups from {event}",
        "pull together context from {event}",
        "what was discussed during {event}",
        "show me the action items from {event}",
        "what changed after {event}",
        "find the relevant details from {event}",
        "what did we cover in {event} about {topic}",
        "find decisions from {event} related to {topic}",
        "what follow ups came out of {event} for {project}",
        "summarize the discussion about {topic} during {event}",
        "summarize the {project} meeting",
        "what happened around {time_window}",
    ],
    "hasDeepResearchIntent": [
        "do deep research on {topic}",
        "give me a thorough analysis of {topic}",
        "look into {topic} in depth",
        "research {topic} and compare the evidence",
        "build me a detailed research brief on {topic}",
        "investigate {topic} from primary sources",
        "find everything important about {topic}",
        "deep dive into {topic}",
        "prepare a research dossier on {topic}",
        "analyze the current state of {topic}",
    ],
    "hasInsightIntent": [
        "what's interesting about {topic}",
        "give me insight into {topic}",
        "tell me something useful about {topic}",
        "what patterns do you see in {topic}",
        "summarize the key insight from {topic}",
        "what should i notice about {topic}",
        "explain the important takeaway from {topic}",
        "what does {topic} imply",
        "find the signal in {topic}",
        "help me understand {topic}",
    ],
    "hasBrowsingAgentIntent": [
        "browse {site} for {query}",
        "navigate to {site} and find {query}",
        "open {site} and look up {query}",
        "go to {site} and search for {query}",
        "check {site} for {topic}",
        "use the browser to find {query}",
        "look up {query} on {site}",
        "open the web and compare {topic}",
        "find the latest page about {topic}",
        "search online for {query}",
    ],
    "hasCallingAgentIntent": [
        "call {contact}",
        "start a call with {contact}",
        "phone {contact} about {call_reason}",
        "dial {contact}",
        "place a call to {contact}",
        "set up a call with {contact} about {call_reason}",
        "connect me to {contact}",
        "call the {service_contact}",
        "ring {contact} and ask about {call_reason}",
        "start a voice call for {call_reason}",
    ],
}

SLOT_VALUES = {
    "subject": [
        "the receipt",
        "my passport",
        "the whiteboard",
        "the parking sign",
        "my dog at the beach",
        "the restaurant menu",
        "the hotel lobby",
        "the concert ticket",
        "the red bike",
        "our team dinner",
        "the handwritten note",
        "the license plate",
        "the mountain view",
        "the birthday cake",
        "the slide with revenue numbers",
    ],
    "time_window": [
        "yesterday",
        "last weekend",
        "last summer",
        "this morning",
        "january",
        "my trip to berlin",
        "the conference",
        "two weeks ago",
        "last christmas",
        "the product launch",
    ],
    "topic": [
        "battery degradation",
        "edge ai inference",
        "the vendor contract",
        "customer churn",
        "eu privacy rules",
        "on device speech models",
        "quarterly hiring plans",
        "new laptop options",
        "local restaurant permits",
        "the market for home robots",
        "synthetic data quality",
        "meeting follow ups",
        "health insurance options",
        "open source tts licensing",
        "android cpu acceleration",
        "retention by cohort",
        "the school district budget",
        "travel restrictions",
        "the new camera sensor",
        "semantic parsing datasets",
    ],
    "place": [
        "vienna",
        "the office",
        "the airport",
        "the hotel",
        "new york",
        "the kitchen",
        "the conference room",
        "the train station",
        "the museum",
        "the workshop",
    ],
    "project": [
        "inherent export",
        "the launch plan",
        "the hiring packet",
        "the q three review",
        "the board memo",
        "the android integration",
        "the user study",
        "the training run",
    ],
    "document_type": [
        "project brief",
        "design doc",
        "status report",
        "launch checklist",
        "research memo",
        "meeting summary",
        "incident report",
        "proposal",
    ],
    "event": [
        "the monday sync",
        "the design review",
        "the customer call",
        "the budget meeting",
        "the planning session",
        "the interview loop",
    ],
    "site": [
        "wikipedia",
        "github",
        "the company site",
        "amazon",
        "maps",
        "youtube",
        "the docs",
        "hacker news",
        "the support portal",
        "the government website",
    ],
    "query": [
        "battery replacement",
        "nearest pharmacy",
        "refund policy",
        "conference schedule",
        "train tickets",
        "api documentation",
        "best reviews",
        "price history",
        "weather tomorrow",
        "recent research",
        "privacy policy",
        "setup instructions",
    ],
    "contact": [
        "alex",
        "sam",
        "jordan",
        "morgan",
        "casey",
        "taylor",
        "the office",
        "my manager",
        "the supplier",
        "the hotel",
        "the restaurant",
        "customer support",
        "the pharmacy",
        "the clinic",
        "the airline",
        "the school",
        "the contractor",
        "the delivery driver",
        "the bank",
        "the dealership",
    ],
    "service_contact": [
        "doctor's office",
        "support desk",
        "front desk",
        "pharmacy",
        "airline desk",
        "bank branch",
        "insurance office",
        "delivery company",
        "repair shop",
        "restaurant",
    ],
    "call_reason": [
        "the appointment",
        "the reservation",
        "the refund",
        "the delivery",
        "the invoice",
        "the contract",
        "the meeting",
        "the interview",
        "the prescription",
        "the booking",
        "the repair",
        "the order",
        "the claim",
        "the schedule",
        "the pickup",
    ],
}

PROMPT_WRAPPERS = (
    "{base}",
    "please {base}",
    "can you {base}",
    "could you {base}",
    "i need you to {base}",
    "help me {base}",
    "go ahead and {base}",
    "hey assistant {base}",
    "when you can {base}",
    "for me {base}",
    "{base} for me",
    "{base} right now",
    "{base} please",
    "quickly {base}",
    "would you {base}",
    "i want to {base}",
    "let's {base}",
    "start by trying to {base}",
    "now {base}",
    "on this device {base}",
    "using my context {base}",
    "from my recent activity {base}",
    "based on what i was doing {base}",
    "with the current context {base}",
    "while i'm working {base}",
    "before i forget {base}",
    "if possible {base}",
    "as soon as you can {base}",
    "in the background {base}",
    "without opening anything else {base}",
    "using the assistant {base}",
    "for this task {base}",
    "for later {base}",
    "and keep it ready {base}",
    "then show me the result {base}",
    "and save the result {base}",
    "and make it easy to review {base}",
    "with a concise answer {base}",
    "with enough detail {base}",
    "from the last few days {base}",
    "from this week {base}",
    "from my work account {base}",
    "from my personal context {base}",
    "for the meeting {base}",
    "for my notes {base}",
    "for tomorrow {base}",
    "for the project {base}",
    "using the latest information {base}",
    "with sources {base}",
    "and compare options {base}",
    "and summarize the tradeoffs {base}",
    "then make a short summary {base}",
    "in a way i can act on {base}",
    "for the current conversation {base}",
    "while keeping it private {base}",
    "and tell me when it is ready {base}",
    "with no extra setup {base}",
    "using the default app {base}",
    "as a quick pass {base}",
    "as a detailed pass {base}",
    "for a follow up {base}",
    "for this afternoon {base}",
    "for the next meeting {base}",
    "and include the important details {base}",
    "and keep the answer short {base}",
    "with a practical recommendation {base}",
    "as a background task {base}",
    "after checking my context {base}",
    "from the relevant app {base}",
    "using my saved information {base}",
    "without sharing private details {base}",
    "and ask before taking action {base}",
    "then confirm the next step {base}",
    "and prepare the result for review {base}",
)


def expand_prompts(head: str, count: int) -> list[str]:
    """Deterministically expand prompt templates for a synthetic intent head."""
    if head not in SYNTHETIC_HEADS:
        raise ValueError(f"head {head!r} is not a TTS-only synthetic head")
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")

    templates = PROMPT_TEMPLATES_BY_HEAD[head]
    prompts: list[str] = []
    seen: set[str] = set()
    for template in templates:
        for prompt in _expand_template(template):
            for wrapped_prompt in _wrap_prompt(prompt):
                if wrapped_prompt in seen:
                    continue
                seen.add(wrapped_prompt)
                prompts.append(wrapped_prompt)
                if len(prompts) == count:
                    return prompts
    raise ValueError(
        f"only {len(prompts)} unique prompts available for {head}; requested {count}"
    )


def synthesize(
    prompts: Sequence[str],
    head: str,
    output_dir: Path,
    voices: Sequence[str] = DEFAULT_VOICES,
    *,
    tts_engine: str = OPENF5_TTS_ENGINE,
    mic_augment: bool | None = None,
    mic_snr_db_range: tuple[float, float] | list[float] | None = None,
) -> list[SyntheticSample]:
    return list(
        iter_synthesize(
            prompts,
            head,
            output_dir,
            voices=voices,
            tts_engine=tts_engine,
            mic_augment=mic_augment,
            mic_snr_db_range=mic_snr_db_range,
        )
    )


def iter_synthesize(
    prompts: Sequence[str],
    head: str,
    output_dir: Path,
    voices: Sequence[str] = DEFAULT_VOICES,
    runtime: object | None = None,
    *,
    tts_engine: str = OPENF5_TTS_ENGINE,
    mic_augment: bool | None = None,
    mic_snr_db_range: tuple[float, float] | list[float] | None = None,
) -> Iterator[SyntheticSample]:
    """Render prompts and return manifest rows.

    OpenF5 (`openf5-tts`): voice refs under data/tts_voices/<voice_id>/ref.{wav,txt}.
    Supertonic (`supertonic-3`): preset voices via SUPERTONIC_VOICE_BY_ID in tts_engines.py.
    """
    if head not in SYNTHETIC_HEADS:
        raise ValueError(f"head {head!r} is not a TTS-only synthetic head")
    if head not in INTENT_HEAD_ORDER:
        raise ValueError(f"head {head!r} is not an intent head")
    if not prompts:
        raise ValueError("cannot synthesize an empty prompt list")
    if not voices:
        raise ValueError("at least one voice id is required")
    if tts_engine not in SUPPORTED_TTS_ENGINES:
        raise ValueError(f"unsupported tts_engine {tts_engine!r}")
    if runtime is None:
        from .tts_engines import create_tts_runtime

        runtime = create_tts_runtime(tts_engine)
    use_mic_augment = mic_augment_enabled() if mic_augment is None else mic_augment
    snr_range = parse_snr_db_range(mic_snr_db_range)

    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    voice_root = _voice_root() if tts_engine == OPENF5_TTS_ENGINE else None
    if tts_engine == OPENF5_TTS_ENGINE:
        _require_openf5_cli()
    total = len(prompts) * len(voices)
    completed = 0
    for voice_id in voices:
        ref_audio: Path | None = None
        ref_text: str | None = None
        if voice_root is not None:
            ref_audio, ref_text = _voice_reference(voice_root, voice_id)
        for index, prompt in enumerate(prompts):
            normalized_prompt = _normalize_prompt(prompt)
            wav_path = _synthesize_one(
                prompt=normalized_prompt,
                head=head,
                voice_id=voice_id,
                ref_audio=ref_audio,
                ref_text=ref_text,
                output_root=output_root,
                index=index,
                runtime=runtime,
                tts_engine=tts_engine,
                mic_augment=use_mic_augment,
                mic_snr_db_range=snr_range,
            )
            sample = SyntheticSample(
                audio_path=wav_path,
                transcript=normalized_prompt,
                head=head,
                voice_id=voice_id,
                tts_engine=tts_engine,
            )
            completed += 1
            if total > 1 and (completed == 1 or completed % 100 == 0 or completed == total):
                print(f"synthesized {completed}/{total} samples for {head}", flush=True)
            yield sample


def write_synthetic_manifest(samples: Sequence[SyntheticSample], output_path: str | Path) -> int:
    if not samples:
        raise ValueError("cannot write an empty synthetic manifest")
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["audio_path", "transcript", "head", "voice_id", "tts_engine"],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "audio_path": str(sample.audio_path.resolve()),
                    "transcript": sample.transcript,
                    "head": sample.head,
                    "voice_id": sample.voice_id,
                    "tts_engine": sample.tts_engine,
                }
            )
    return len(samples)


def _expand_template(template: str) -> list[str]:
    slots = [slot for slot in SLOT_VALUES if "{" + slot + "}" in template]
    if not slots:
        return [template]
    prompts = [template]
    for slot in slots:
        values = SLOT_VALUES[slot]
        prompts = [prompt.replace("{" + slot + "}", value) for prompt in prompts for value in values]
    return prompts


def _wrap_prompt(prompt: str) -> list[str]:
    base = _normalize_prompt(prompt)
    return [_normalize_prompt(wrapper.format(base=base)) for wrapper in PROMPT_WRAPPERS]


def _require_openf5_cli() -> None:
    if _openf5_command() is None:
        raise RuntimeError(f"{OPENF5_TTS_COMMAND} is required for OpenF5 synthetic TTS generation")


def _voice_root() -> Path:
    return Path(os.environ.get("INHERENT_TTS_VOICE_DIR", str(DEFAULT_TTS_VOICE_DIR))).expanduser()


def _voice_reference(voice_root: Path, voice_id: str) -> tuple[Path, str]:
    if voice_id.strip() == "":
        raise ValueError("voice id must be non-empty")
    ref_dir = voice_root / voice_id
    ref_audio = ref_dir / "ref.wav"
    ref_text_path = ref_dir / "ref.txt"
    if not ref_audio.is_file():
        raise FileNotFoundError(f"missing TTS reference audio: {ref_audio}")
    if not ref_text_path.is_file():
        raise FileNotFoundError(f"missing TTS reference transcript: {ref_text_path}")
    ref_text = ref_text_path.read_text().strip()
    if not ref_text:
        raise ValueError(f"TTS reference transcript is empty: {ref_text_path}")
    return ref_audio, ref_text


def _synthesize_one(
    *,
    prompt: str,
    head: str,
    voice_id: str,
    ref_audio: Path | None,
    ref_text: str | None,
    output_root: Path,
    index: int,
    runtime: object,
    tts_engine: str,
    mic_augment: bool = False,
    mic_snr_db_range: tuple[float, float] = (10.0, 22.0),
) -> Path:
    digest = hashlib.sha1(f"{head}:{voice_id}:{index}:{prompt}".encode()).hexdigest()[:16]
    sample_dir = output_root / head / voice_id / digest
    final_path = output_root / head / voice_id / f"{index:08d}_{digest}.wav"
    if final_path.is_file():
        return final_path
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True)
    generated_path = sample_dir / "generated.wav"
    if tts_engine == OPENF5_TTS_ENGINE:
        if ref_audio is None or ref_text is None:
            raise ValueError("OpenF5 synthesis requires reference audio and text")
        runtime.synthesize_to_wav(  # type: ignore[union-attr]
            prompt=prompt,
            ref_audio=ref_audio,
            ref_text=ref_text,
            output_path=generated_path,
        )
        if not generated_path.is_file():
            raise RuntimeError(f"OpenF5 did not write expected output: {generated_path}")
        normalized_path = sample_dir / "normalized.wav"
        _normalize_wav(generated_path, normalized_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_path.replace(final_path)
    elif tts_engine == SUPERTONIC_TTS_ENGINE:
        runtime.synthesize_to_wav(prompt=prompt, voice_id=voice_id, output_path=final_path)  # type: ignore[union-attr]
        if not final_path.is_file():
            raise RuntimeError(f"supertonic did not write expected output: {final_path}")
    else:
        raise ValueError(f"unsupported tts_engine {tts_engine!r}")
    shutil.rmtree(sample_dir, ignore_errors=True)
    if mic_augment:
        seed = int(hashlib.sha1(f"{head}:{voice_id}:{index}:{prompt}:mic".encode()).hexdigest()[:8], 16)
        augment_wav_file(final_path, snr_db_range=mic_snr_db_range, seed=seed)
    return final_path


class _OpenF5Runtime:
    engine = OPENF5_TTS_ENGINE
    def __init__(self, model_files: OpenF5ModelFiles) -> None:
        try:
            from f5_tts.infer.utils_infer import load_model, load_vocoder
            from hydra.utils import get_class
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise RuntimeError("f5-tts and its runtime dependencies are required for synthetic TTS") from exc

        self.device = _openf5_device()
        self.model_cfg = OmegaConf.load(str(model_files.model_cfg))
        mel_spec = self.model_cfg.model.mel_spec
        if int(mel_spec.target_sample_rate) != 24_000:
            raise ValueError(f"OpenF5 target_sample_rate must be 24000, got {mel_spec.target_sample_rate}")
        if int(mel_spec.n_mel_channels) != 100:
            raise ValueError(f"OpenF5 n_mel_channels must be 100, got {mel_spec.n_mel_channels}")

        self.mel_spec_type = str(mel_spec.mel_spec_type)
        self.vocoder = load_vocoder(vocoder_name=self.mel_spec_type, is_local=False, local_path="", device=self.device)
        model_cls = get_class(f"f5_tts.model.{self.model_cfg.model.backbone}")
        self.model = load_model(
            model_cls,
            self.model_cfg.model.arch,
            str(model_files.ckpt_file),
            mel_spec_type=self.mel_spec_type,
            vocab_file=str(model_files.vocab_file),
            device=self.device,
        )
        self._ref_cache: dict[tuple[Path, str], tuple[object, int, str]] = {}

    def synthesize_to_wav(
        self,
        *,
        prompt: str,
        ref_audio: Path,
        ref_text: str,
        output_path: Path,
    ) -> None:
        try:
            import soundfile as sf
            import torchaudio
            from f5_tts.infer.utils_infer import (
                infer_batch_process,
                preprocess_ref_audio_text,
                remove_silence_for_generated_wav,
            )
        except ImportError as exc:
            raise RuntimeError("f5-tts, soundfile, and torchaudio are required for synthetic TTS") from exc

        ref_key = (ref_audio.resolve(), ref_text)
        if ref_key not in self._ref_cache:
            processed_audio, processed_text = preprocess_ref_audio_text(
                str(ref_audio.resolve()),
                ref_text,
                show_info=lambda _message: None,
            )
            audio, sample_rate = torchaudio.load(processed_audio)
            self._ref_cache[ref_key] = (audio, int(sample_rate), processed_text)

        audio, sample_rate, processed_text = self._ref_cache[ref_key]
        result = next(
            infer_batch_process(
                (audio, sample_rate),
                processed_text,
                [prompt],
                self.model,
                self.vocoder,
                mel_spec_type=self.mel_spec_type,
                progress=None,
                device=self.device,
                nfe_step=16,
            )
        )
        generated_wave, final_sample_rate, _spectrogram = result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if int(final_sample_rate) != SAMPLE_RATE:
            from scipy.signal import resample_poly
            generated_wave = resample_poly(generated_wave, SAMPLE_RATE, int(final_sample_rate)).astype(generated_wave.dtype)
            final_sample_rate = SAMPLE_RATE
        sf.write(str(output_path), generated_wave, final_sample_rate)
        remove_silence_for_generated_wav(str(output_path))


def _openf5_device() -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for OpenF5 synthetic TTS generation") from exc
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _normalize_wav(input_path: Path, output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to normalize synthetic TTS WAV files")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-sample_fmt",
            "s16",
            str(output_path),
        ],
        check=True,
    )
    if not output_path.is_file():
        raise RuntimeError(f"ffmpeg did not write normalized output: {output_path}")


def _openf5_command() -> str | None:
    command = shutil.which(OPENF5_TTS_COMMAND)
    if command is not None:
        return command
    sibling = Path(sys.executable).with_name(OPENF5_TTS_COMMAND)
    if sibling.is_file():
        return str(sibling)
    return None


def _openf5_model_reference() -> str:
    raw_value = os.environ.get(OPENF5_MODEL_ENV, "").strip()
    if not raw_value:
        raise RuntimeError(
            f"{OPENF5_MODEL_ENV} must be set to mrfakename/OpenF5-TTS-Base "
            "or a vetted local OpenF5 model path"
        )
    return _validate_openf5_model_reference(raw_value)


def _openf5_model_files() -> OpenF5ModelFiles:
    model_ref = _openf5_model_reference()
    model_path = Path(model_ref).expanduser()
    if not model_path.exists():
        model_path = Path(_download_openf5_repo(model_ref))
    if not model_path.is_dir():
        raise ValueError(f"{OPENF5_MODEL_ENV} must resolve to a directory with OpenF5 model files: {model_path}")
    files = OpenF5ModelFiles(
        model_cfg=model_path / "config.yaml",
        ckpt_file=model_path / "model.pt",
        vocab_file=model_path / "vocab.txt",
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"OpenF5 model directory is missing required files: {missing}")
    return files


def _download_openf5_repo(repo_id: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download OpenF5 model files") from exc
    return snapshot_download(
        repo_id=repo_id,
        allow_patterns=("config.yaml", "model.pt", "vocab.txt"),
    )


def _validate_openf5_model_reference(value: str) -> str:
    normalized = value.lower()
    disallowed = {model.lower() for model in DISALLOWED_TTS_MODEL_IDS}
    if normalized in disallowed or "xtts" in normalized or normalized.endswith("/f5-tts"):
        raise ValueError(
            f"disallowed TTS model {value!r}; use mrfakename/OpenF5-TTS-Base "
            "or a vetted local OpenF5 model path"
        )
    local_path = Path(value).expanduser()
    if local_path.exists():
        return str(local_path.resolve())
    if value in APPROVED_OPENF5_MODEL_IDS:
        return value
    raise ValueError(
        f"{OPENF5_MODEL_ENV} must name an approved OpenF5 model "
        f"{APPROVED_OPENF5_MODEL_IDS} or an existing local model path; got {value!r}"
    )


def _normalize_prompt(prompt: str) -> str:
    normalized = " ".join(prompt.lower().strip().split())
    if not normalized:
        raise ValueError("prompt must be non-empty")
    return normalized
