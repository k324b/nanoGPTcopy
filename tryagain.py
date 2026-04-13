#!/usr/bin/env python3
import time
import numpy as np
import speech_recognition as sr
from faster_whisper import WhisperModel
import roslibpy

# -----------------------------
# Connect to Duckiebot rosbridge
# -----------------------------
client = roslibpy.Ros(host='entebot208.local', port=9001)
client.run()

topic = roslibpy.Topic(
    client,
    '/entebot208/car_cmd_switch_node/cmd',
    'duckietown_msgs/Twist2DStamped'
)

# -----------------------------
# Whisper STT
# -----------------------------
stt = WhisperModel("tiny.en", compute_type="int8")

def normalize(text): return text.lower().strip()

def detect(text):
    text = normalize(text)
    if "forward" in text: return "forward"
    if "backward" in text: return "backward"
    if "left" in text: return "left"
    if "right" in text: return "right"
    if "stop" in text: return "stop"
    return None

def send_command(command):
    v, omega = 0.0, 0.0
    if command == "forward":   v = 0.3
    elif command == "backward": v = -0.3
    elif command == "left":    v, omega = 0.2, 2.0
    elif command == "right":   v, omega = 0.2, -2.0

    topic.publish(roslibpy.Message({
        'header': {'stamp': {'secs': 0, 'nsecs': 0}, 'frame_id': ''},
        'v': v,
        'omega': omega
    }))
    print(f"Command: {command} | v={v}, omega={omega}")

# -----------------------------
# Main loop
# -----------------------------
recognizer = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    print("Calibrating microphone...")
    recognizer.adjust_for_ambient_noise(source, duration=1.0)

print("Listening for commands...")

try:
    while client.is_connected:
        with mic as source:
            print("Listening...")
            audio = recognizer.listen(source, timeout=5.0, phrase_time_limit=3.0)

        audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        segments, _ = stt.transcribe(audio_np)
        text = " ".join(s.text for s in segments).strip()

        if not text:
            continue

        print(f"Heard: {text}")
        command = detect(text)

        if command:
            send_command(command)
            time.sleep(0.2)
        else:
            print("No command detected.")

except KeyboardInterrupt:
    print("Shutting down...")
    send_command("stop")
finally:
    topic.unadvertise()
    client.terminate()
