#!/usr/bin/env -S uv run --script --quiet

"""
    A python script that record a 5sec video from the webcam.
"""

import argparse
import sys
import subprocess
import requests
import mimetypes
import threading
from datetime import datetime

def record_video(video_file_name):
    #Linux command = ffmpeg -f pulse -ac 2 -i default -f v4l2 -i /dev/video0 -t 00:00:05 -vcodec libx264 record.mp4

    cmd = [
    "ffmpeg",
    "-f", "pulse",
    "-ac", "2",
    "-i", "default",
    "-f", "v4l2",
    "-i", "/dev/video0",
    "-t", "00:00:05",
    "-vcodec", "libx264",
    video_file_name
    ]
    
    print("coucou")
    try:
        subprocess.run(cmd, check=True)
        print("[VIDEO] Video was successfully saved")
    except subprocess.CalledProcessError:
        print("[VIDEO] Error, couln't save the video.")

def record_audio(audio_file_name):

    cmd = [
        "ffmpeg",
        "-f", "alsa",
        "-i", "default",
        "-t", "5",
        "-ac", "1",
        "-ar", "16000",
        "-y",
        audio_file_name
    ]

    try:
        subprocess.run(cmd, check=True)
        print("[AUDIO] Audio was successfully saved")
    except subprocess.CalledProcessError:
        print("[AUDIO] Error, couln't save the audio.")

def send_video_and_audio_to_API(video_file_name, audio_file_name):
    url = "http://0.0.0.0:8000/process"

    # Déterminer le type MIME de chaque fichier
    video_mime, _ = mimetypes.guess_type(video_file_name)
    audio_mime, _ = mimetypes.guess_type(audio_file_name)

    # Ouvrir les deux fichiers en mode binaire
    with open(video_file_name, "rb") as vf, open(audio_file_name, "rb") as af:
        files = {
            "video": (video_file_name, vf, video_mime or "application/octet-stream"),
            "audio": (audio_file_name, af, audio_mime or "application/octet-stream")
        }

        r = requests.post(url, files=files)
        r.raise_for_status()

    result = r.json()

    return result

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_file_name = f"video_{timestamp}.mp4"
    audio_file_name = f"audio_{timestamp}.wav"

    # Record video + audio in parallele
    t_video = threading.Thread(target=record_video, args=(video_file_name,))
    t_audio = threading.Thread(target=record_audio, args=(audio_file_name,))

    print("TOP DEPART")
    t_video.start()
    t_audio.start()

    # Attendre que les deux soient finis
    t_video.join()
    t_audio.join()

    #Send the video and audio to Harmoni
    reply = send_video_and_audio_to_API(video_file_name, audio_file_name)

    print(f"User: {reply['transcript']}")
    print(f"\nAssistant: {reply['answer']}")

if __name__ == '__main__':
    main()