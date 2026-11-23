"""
Standalone script to benchmark detect_speaking_face in isolation.

Usage:
    python detect_speaker.py video.mp4
    python detect_speaker.py video.mp4 --audio audio.wav
    python detect_speaker.py videos/               # process a whole directory
    python detect_speaker.py video.mp4 --stride 5   # override FRAME_STRIDE
    python detect_speaker.py video.mp4 --no-save     # skip saving frame images

Runs from the .testFiles/ directory (or anywhere) as long as the project root
is one level up.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Add project root so we can import vision/ and config/
sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".mpeg", ".mpg"}


def resolve_path(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (SCRIPT_DIR / p).resolve()
    return p


def video_info(path: Path) -> dict:
    """Return basic metadata for a video file."""
    cap = cv2.VideoCapture(str(path))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def run_detection(video_path: Path, audio_path: Path | None, save_frames: bool, stride_override: int | None):
    """Run detect_speaking_face on a single video and print timing breakdown."""
    # Lazy imports so argparse errors show before slow model loading
    from config.settings import FRAME_STRIDE
    import config.settings as settings

    if stride_override is not None:
        settings.FRAME_STRIDE = stride_override
        effective_stride = stride_override
    else:
        effective_stride = FRAME_STRIDE

    from vision.detection import detect_speaking_face

    info = video_info(video_path)

    print(f"\n{'='*60}")
    print(f"Video     : {video_path.name}")
    print(f"Resolution: {info['width']}x{info['height']} @ {info['fps']:.1f} fps")
    print(f"Duration  : {info['duration']:.2f}s  ({info['total_frames']} frames)")
    print(f"Stride    : every {effective_stride} frame(s)  →  ~{info['total_frames'] // effective_stride} processed")
    if audio_path:
        print(f"Audio     : {audio_path.name} (VAD enabled)")
    else:
        print(f"Audio     : none (VAD disabled)")
    print(f"{'='*60}")

    audio_str = str(audio_path) if audio_path else None

    start = time.time()
    speaking_faces, face_grid, speaking_probs = detect_speaking_face(
        str(video_path),
        audio_path=audio_str,
        save_frames=save_frames,
        verbose=True,
    )
    elapsed = time.time() - start

    num_faces = face_grid.shape[1]
    speaker_idx = speaking_probs.index(max(speaking_probs)) if speaking_probs else -1

    print(f"\n--- Results ---")
    print(f"Faces detected : {num_faces}")
    for i, prob in enumerate(speaking_probs):
        marker = "  ◀ SPEAKER" if i == speaker_idx else ""
        print(f"  Face {i}: speaking prob = {prob:.4f}{marker}")
    print(f"Speaking frames: {len(speaking_faces)}")
    print(f"\nTotal time: {elapsed:.4f}s")
    print(f"{'='*60}\n")

    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark detect_speaking_face on video file(s).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python detect_speaker.py video.mp4\n"
            "  python detect_speaker.py video.mp4 --audio audio.wav\n"
            "  python detect_speaker.py videos/\n"
            "  python detect_speaker.py video.mp4 --stride 5\n"
            "  python detect_speaker.py video.mp4 --no-save\n"
        ),
    )

    parser.add_argument("path", help="Path to a video file or directory of videos.")
    parser.add_argument("--audio", default=None, help="Path to a separate audio file (enables VAD).")
    parser.add_argument("--stride", type=int, default=None, help="Override FRAME_STRIDE (default: from config).")
    parser.add_argument("--no-save", action="store_true", help="Skip saving detected face frames to disk.")

    args = parser.parse_args()
    save_frames = not args.no_save

    target = resolve_path(args.path)
    audio_path = resolve_path(args.audio) if args.audio else None

    if audio_path and not audio_path.is_file():
        parser.error(f"Audio file not found: {audio_path}")

    if target.is_file():
        if target.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            parser.error(
                f"Unsupported video type '{target.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}"
            )
        video_files = [target]
    elif target.is_dir():
        video_files = sorted(
            f for f in target.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        )
        if not video_files:
            parser.error(f"No supported video files found in: {target}")
    else:
        parser.error(f"Path not found: {target}")

    print(f"Loading models...")
    load_start = time.time()
    # Trigger model loading by importing detection (FaceAnalysis init happens at import)
    import vision.detection  # noqa: F401
    load_elapsed = time.time() - load_start
    print(f"Models loaded in {load_elapsed:.2f}s")

    timings = []
    for vf in video_files:
        elapsed = run_detection(vf, audio_path, save_frames, args.stride)
        timings.append((vf.name, elapsed))

    if len(timings) > 1:
        print(f"\n{'='*60}")
        print("Summary:")
        for name, t in timings:
            print(f"  {name:40s}  {t:.4f}s")
        total = sum(t for _, t in timings)
        print(f"  {'Total':40s}  {total:.4f}s")
        print(f"  {'Average':40s}  {total / len(timings):.4f}s")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
