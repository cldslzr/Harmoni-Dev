"""
Simple CLI to send a video and audio file to the HARMONI API and print the response.

Usage:
    python video_terminal/video_terminal.py -v path/to/video.mp4 -a path/to/audio.wav
    python video_terminal/video_terminal.py -v path/to/video.mp4  # audio extracted from video by the API
    python video_terminal/video_terminal.py -v path/to/video.mp4 --speak  # enable real-time TTS
"""

import argparse
import queue
import re
import sys
import os
import json
import threading
import requests

# ── TTS configuration ──────────────────────────────────────────────────────────
# Change SPEAK_LANG to switch the TTS language.
#   'fr' → French          (requires: espeak-ng for G2P)
#   'en' → American English
# Dependencies: pip install kokoro sounddevice
# Linux:        sudo apt-get install espeak-ng
SPEAK_LANG = 'fr'

_TTS_CONFIG = {
    'fr': {'lang_code': 'f', 'voice': 'ff_siwis'},
    'en': {'lang_code': 'a', 'voice': 'af_heart'},
}
# ──────────────────────────────────────────────────────────────────────────────

BASE_URL = "http://0.0.0.0:8000"


class SpeechPlayer:
    """
    Buffers streamed LLM tokens and feeds Kokoro TTS in a background thread,
    so audio starts playing as soon as the first natural sentence boundary
    arrives — without blocking the token stream.

    Chunking strategy
    -----------------
    • Flush on sentence-ending punctuation (. ! ? …) when ≥ MIN_WORDS_FOR_FLUSH
      words have accumulated  →  low latency, natural prosody.
    • Flush unconditionally at MAX_WORDS_HARD_LIMIT words  →  handles very long
      unpunctuated sentences so audio never lags too far behind text.

    Usage
    -----
        player = SpeechPlayer()
        for token in stream:
            player.push_token(token)
        player.finish()   # flush remainder and wait for playback to end
    """

    _SENTENCE_END_RE = re.compile(r'[.!?…]+\s*$')
    _MIN_WORDS_FOR_FLUSH = 5    # don't flush very short fragments on punctuation
    _MAX_WORDS_HARD_LIMIT = 25  # hard flush regardless of punctuation

    def __init__(self, lang: str = SPEAK_LANG):
        cfg = _TTS_CONFIG.get(lang, _TTS_CONFIG['en'])
        try:
            from kokoro import KPipeline
            import sounddevice as _sd
        except ImportError as exc:
            print(
                f"\n[TTS] Missing dependency: {exc}\n"
                "  Install with:  pip install kokoro sounddevice\n"
                "  On Linux also: sudo apt-get install espeak-ng",
                file=sys.stderr,
            )
            raise

        self._sd = _sd
        self._voice = cfg['voice']

        print("[TTS] Loading Kokoro...", end=" ", flush=True)
        self._pipeline = KPipeline(lang_code=cfg['lang_code'])
        print("ready.", flush=True)

        self._buffer: list[str] = []
        self._chunk_queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._playback_worker, daemon=True)
        self._worker.start()

    def push_token(self, token: str) -> None:
        """Add one token from the LLM stream; flushes to TTS on natural boundaries."""
        self._buffer.append(token)
        buf_text = ''.join(self._buffer)
        word_count = len(buf_text.split())

        sentence_done = self._SENTENCE_END_RE.search(buf_text)
        if (sentence_done and word_count >= self._MIN_WORDS_FOR_FLUSH) \
                or word_count >= self._MAX_WORDS_HARD_LIMIT:
            self._flush_buffer()

    def finish(self) -> None:
        """Flush any remaining buffered text and block until all audio has played."""
        self._flush_buffer()
        self._chunk_queue.put(None)  # sentinel
        self._worker.join()

    # ── private helpers ──────────────────────────────────────────────────────

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        chunk = ''.join(self._buffer).strip()
        self._buffer.clear()
        if chunk:
            self._chunk_queue.put(chunk)

    def _playback_worker(self) -> None:
        """Background thread: synthesize each chunk and play it immediately."""
        while True:
            chunk = self._chunk_queue.get()
            if chunk is None:
                break
            try:
                for _, _, audio in self._pipeline(chunk, voice=self._voice):
                    if audio is not None:
                        self._sd.play(audio.numpy(), samplerate=24000)
                        self._sd.wait()
            except Exception as exc:
                print(f"\n[TTS error] {exc}", file=sys.stderr)



def process_video_stream(video_path, audio_path=None, speaker: SpeechPlayer = None):
    """Send video (and optionally audio) to /process_stream and print the response as it streams."""
    url = f"{BASE_URL}/process_stream"
    with open(video_path, "rb") as vf:
        files = {"video": ("video.mp4", vf, "video/mp4")}
        if audio_path:
            with open(audio_path, "rb") as af:
                files["audio"] = ("audio.wav", af, "audio/wav")
                response = requests.post(url, files=files, stream=True)
        else:
            response = requests.post(url, files=files, stream=True)

    if response.status_code == 422:
        detail = response.json().get("detail", "Unprocessable video")
        print(f"Error: {detail}", file=sys.stderr)
        sys.exit(1)
    response.raise_for_status()

    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        event = json.loads(raw_line)
        if event["type"] == "info":
            print(f"Detected user: {event['detected_user']}")
            print(f"Emotion: {event['emotion']}")
            print(f"Transcript: {event['transcript']}")
            print(f"\nAssistant: ", end="", flush=True)
        elif event["type"] == "token":
            content = event["content"]
            print(content, end="", flush=True)
            if speaker is not None:
                speaker.push_token(content)
        elif event["type"] == "done":
            print()  # final newline after the streamed answer
            if speaker is not None:
                speaker.finish()


def process_video(video_path, audio_path=None):
    """Send video (and optionally audio) to the HARMONI /process endpoint."""
    url = f"{BASE_URL}/process"
    with open(video_path, "rb") as vf:
        files = {
            "video": ("video.mp4", vf, "video/mp4"),
        }
        if audio_path:
            with open(audio_path, "rb") as af:
                files["audio"] = ("audio.wav", af, "audio/wav")
                response = requests.post(url, files=files)
        else:
            response = requests.post(url, files=files)

    if response.status_code == 422:
        detail = response.json().get("detail", "Unprocessable video")
        print(f"Error: {detail}", file=sys.stderr)
        sys.exit(1)
    response.raise_for_status()

    payload = response.json()
    print(f"Detected user: {payload.get('detected_user', '')}")
    print(f"Emotion: {payload.get('emotion', '')}")
    print(f"Transcript: {payload.get('transcript', '')}")
    print(f"\nAssistant: {payload.get('answer', '')}")
    print(payload.get("response", ""))


def main():
    parser = argparse.ArgumentParser(description="Send video/audio to HARMONI and get a response.")
    parser.add_argument("-v", "--video", required=True, help="Path to the video file.")
    parser.add_argument("-a", "--audio", default=None, help="Path to the audio file (optional, extracted from video if omitted).")
    parser.add_argument("-s", "--stream", action="store_true", default=False, help="Stream the response instead of waiting for the full response.")
    parser.add_argument("--speak", action="store_true", default=False, help=f"Enable real-time TTS via Kokoro (language: {SPEAK_LANG}). Implies --stream.")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    if args.audio and not os.path.isfile(args.audio):
        print(f"Error: audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    # --speak implies streaming mode
    use_stream = args.stream or args.speak

    speaker = None
    if args.speak:
        try:
            speaker = SpeechPlayer(lang=SPEAK_LANG)
        except ImportError:
            sys.exit(1)

    print("Sending video/audio to HARMONI...")
    if use_stream:
        process_video_stream(args.video, args.audio, speaker=speaker)
    else:
        process_video(args.video, args.audio)

if __name__ == "__main__":
    main()
