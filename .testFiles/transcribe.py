import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import whisper

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".mpeg", ".mpga", ".webm"}

SCRIPT_DIR = Path(__file__).resolve().parent
TRANSCRIPT_DIR = SCRIPT_DIR.parent / "media" / "Transcript"

def resolve_path(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (SCRIPT_DIR / p).resolve()
    return p


def load_first_channel(audio_path: Path) -> np.ndarray:
    command = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(audio_path),
        "-af",
        "pan=mono|c0=c0",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(whisper.audio.SAMPLE_RATE),
        "-",
    ]
    output = subprocess.run(command, check=True, capture_output=True).stdout
    return np.frombuffer(output, np.int16).astype(np.float32) / 32768.0


def transcribe_file(model, audio_path: Path, device: str, transcript_dir: Path) -> None:
    audio = load_first_channel(audio_path)
    duration = len(audio) / whisper.audio.SAMPLE_RATE

    print(f"\n--- {audio_path.name} ---")
    print(f"Duration: {duration:.2f}s")

    start = time.time()
    result = model.transcribe(audio, fp16=(device in ["mps", "cuda"]))
    elapsed = time.time() - start

    transcript_path = transcript_dir / f"{audio_path.stem}.json"
    transcript_path.write_text(
        json.dumps(
            {
                "timestamp": audio_path.stem,
                "locale": result["language"],
                "user": result["text"].strip(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(result["text"].strip())
    print(f"[saved to {transcript_path}]")
    print(f"[transcribed in {elapsed:.2f}s]")


def build_model(device: str, model_name: str):
    print(f"Using device: {device}, model: {model_name}")
    model = whisper.load_model(model_name)
    if device == "mps":
        for module in model.modules():
            for name, buf in list(module.named_buffers(recurse=False)):
                if buf.is_sparse:
                    module.register_buffer(name, buf.to_dense())
        model = model.float()
        model = model.to(device)
    elif device == "cuda":
        model = model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio file(s) with Whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python transcribe.py -c audio/interview.wav\n"
            "  python transcribe.py -m audio/interview.wav\n"
            "  python transcribe.py -c audio/\n"
            "  python transcribe.py -m /absolute/path/to/audio/\n"
        ),
    )

    device_group = parser.add_mutually_exclusive_group(required=True)
    device_group.add_argument("-c", dest="device", action="store_const", const="cpu", help="Use CPU")
    device_group.add_argument("-m", dest="device", action="store_const", const="mps", help="Use MPS (Apple Silicon)")
    device_group.add_argument("-n", dest="device", action="store_const", const="cuda", help="Use CUDA")

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--tiny",   dest="model", action="store_const", const="tiny",   help="Use tiny model")
    model_group.add_argument("--base",   dest="model", action="store_const", const="base",   help="Use base model")
    model_group.add_argument("--small",  dest="model", action="store_const", const="small",  help="Use small model (default)")
    model_group.add_argument("--medium", dest="model", action="store_const", const="medium", help="Use medium model")
    model_group.add_argument("--large",  dest="model", action="store_const", const="large",  help="Use large model")
    model_group.add_argument("--turbo",  dest="model", action="store_const", const="turbo",  help="Use turbo model")

    parser.add_argument("path", help="Path to an audio file or a directory of audio files.")

    args = parser.parse_args()
    if args.model is None:
        args.model = "small"

    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available on this machine. Use -c for CPU instead.")
    elif args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available on this machine. Use -c for CPU instead.")

    target = resolve_path(args.path)

    if target.is_file():
        if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
            parser.error(f"Unsupported file type '{target.suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        audio_files = [target]
    elif target.is_dir():
        audio_files = sorted(
            f for f in target.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not audio_files:
            parser.error(f"No supported audio files found in: {target}")
    else:
        parser.error(f"Path not found: {target}")

    model = build_model(args.device, args.model)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    for audio_file in audio_files:
        transcribe_file(model, audio_file, args.device, TRANSCRIPT_DIR)

    if args.device == "mps":
        torch.mps.empty_cache()
    elif args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
