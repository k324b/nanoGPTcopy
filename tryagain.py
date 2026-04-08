
import rospy
import numpy as np
import speech_recognition as sr
from faster_whisper import WhisperModel
from duckietown_msgs.msg import Twist2DStamped
import time

# -----------------------------
# Initialize Whisper
# -----------------------------
stt = WhisperModel("tiny.en", compute_type="int8")

# -----------------------------
# Normalize text
# -----------------------------
def normalize(text: str) -> str:
    return text.lower().strip()

# -----------------------------
# Detect command
# -----------------------------
def detect(text: str):
    text = normalize(text)

    if "forward" in text:
        return "forward"
    if "backward" in text:
        return "backward"
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    if "stop" in text:
        return "stop"

    return None

# -----------------------------
# ROS Publisher
# -----------------------------
class VoiceController:
    def __init__(self):
        rospy.init_node("voice_control_node")

        self.pub = rospy.Publisher(
            "/entebot208/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1
        )

    def send_command(self, v, omega):
        msg = Twist2DStamped()
        msg.v = v
        msg.omega = omega
        self.pub.publish(msg)

# -----------------------------
# Main Assistant Loop
# -----------------------------
def start_assistant():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    controller = VoiceController()

    print("🎤 Voice control started...")

    while not rospy.is_shutdown():
        try:
            with microphone as source:
                print("Listening...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)

            # Convert audio → numpy
            audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe
            segments, _ = stt.transcribe(audio_np)
            text = " ".join([seg.text for seg in segments]).strip()

            print("Heard:", text)

            command = detect(text)

            # -----------------------------
            # Execute movement
            # -----------------------------
            if command == "forward":
                controller.send_command(0.3, 0.0)

            elif command == "backward":
                controller.send_command(-0.3, 0.0)

            elif command == "left":
                controller.send_command(0.2, 2.0)

            elif command == "right":
                controller.send_command(0.2, -2.0)

            elif command == "stop":
                controller.send_command(0.0, 0.0)

            time.sleep(0.2)

        except Exception as e:
            print(f"Error: {e}")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    start_assistant()
