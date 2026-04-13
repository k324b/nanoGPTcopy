#!/usr/bin/env python3

import time
from typing import Optional

import numpy as np
import speech_recognition as sr
from faster_whisper import WhisperModel

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

# Try Duckietown ROS2 message first; fall back to standard ROS2 Twist
try:
    from duckietown_msgs.msg import Twist2DStamped
    USE_DUCKIETOWN_MSG = True
except ImportError:
    from geometry_msgs.msg import Twist
    USE_DUCKIETOWN_MSG = False


class VoiceControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("voice_control_node")

        # Parameters
        self.declare_parameter("robot_name", "entebot208")
        self.declare_parameter("forward_speed", 0.3)
        self.declare_parameter("backward_speed", -0.3)
        self.declare_parameter("turn_speed", 0.2)
        self.declare_parameter("turn_rate", 2.0)
        self.declare_parameter("listen_timeout", 5.0)
        self.declare_parameter("phrase_time_limit", 3.0)
        self.declare_parameter("command_hold_seconds", 0.2)
        self.declare_parameter("whisper_model", "tiny.en")
        self.declare_parameter("whisper_compute_type", "int8")

        self.robot_name = self.get_parameter("robot_name").value
        self.forward_speed = float(self.get_parameter("forward_speed").value)
        self.backward_speed = float(self.get_parameter("backward_speed").value)
        self.turn_speed = float(self.get_parameter("turn_speed").value)
        self.turn_rate = float(self.get_parameter("turn_rate").value)
        self.listen_timeout = float(self.get_parameter("listen_timeout").value)
        self.phrase_time_limit = float(self.get_parameter("phrase_time_limit").value)
        self.command_hold_seconds = float(self.get_parameter("command_hold_seconds").value)
        self.whisper_model_name = str(self.get_parameter("whisper_model").value)
        self.whisper_compute_type = str(self.get_parameter("whisper_compute_type").value)

        # Topic
        if USE_DUCKIETOWN_MSG:
            topic_name = f"/{self.robot_name}/car_cmd_switch_node/cmd"
            msg_type = Twist2DStamped
            self.get_logger().info(
                f"Using duckietown_msgs/Twist2DStamped on topic: {topic_name}"
            )
        else:
            topic_name = f"/{self.robot_name}/cmd_vel"
            msg_type = Twist
            self.get_logger().warn(
                "duckietown_msgs.msg.Twist2DStamped not found. "
                f"Falling back to geometry_msgs/Twist on topic: {topic_name}"
            )

        qos = QoSProfile(depth=10)
        self.publisher = self.create_publisher(msg_type, topic_name, qos)

        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Faster Whisper
        self.stt = WhisperModel(
            self.whisper_model_name,
            compute_type=self.whisper_compute_type,
        )

        with self.microphone as source:
            self.get_logger().info("Calibrating microphone...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1.0)

        self.get_logger().info("Voice control node started. Listening for commands...")

    @staticmethod
    def normalize(text: str) -> str:
        return text.lower().strip()

    def detect_command(self, text: str) -> Optional[str]:
        text = self.normalize(text)

        if "move forward" in text or "go forward" in text or text == "forward":
            return "forward"
        if "move backward" in text or "go backward" in text or text == "backward":
            return "backward"
        if "turn left" in text or text == "left":
            return "left"
        if "turn right" in text or text == "right":
            return "right"
        if "stop" in text or "halt" in text:
            return "stop"

        return None

    def publish_velocity(self, v: float, omega: float) -> None:
        if USE_DUCKIETOWN_MSG:
            msg = Twist2DStamped()
            # If your ROS2 duckietown_msgs build includes a header, uncomment:
            # msg.header.stamp = self.get_clock().now().to_msg()
            msg.v = float(v)
            msg.omega = float(omega)
        else:
            msg = Twist()
            msg.linear.x = float(v)
            msg.angular.z = float(omega)

        self.publisher.publish(msg)

    def execute_command(self, command: str) -> None:
        if command == "forward":
            self.publish_velocity(self.forward_speed, 0.0)
        elif command == "backward":
            self.publish_velocity(self.backward_speed, 0.0)
        elif command == "left":
            self.publish_velocity(self.turn_speed, self.turn_rate)
        elif command == "right":
            self.publish_velocity(self.turn_speed, -self.turn_rate)
        elif command == "stop":
            self.publish_velocity(0.0, 0.0)

        self.get_logger().info(f"Executed command: {command}")

    def transcribe_once(self) -> Optional[str]:
        with self.microphone as source:
            self.get_logger().info("Listening...")
            audio = self.recognizer.listen(
                source,
                timeout=self.listen_timeout,
                phrase_time_limit=self.phrase_time_limit,
            )

        audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        audio_np = (
            np.frombuffer(audio_data, dtype=np.int16)
            .astype(np.float32) / 32768.0
        )

        segments, _ = self.stt.transcribe(audio_np)
        text = " ".join(segment.text for segment in segments).strip()

        if not text:
            return None

        self.get_logger().info(f"Heard: {text}")
        return text

    def run(self) -> None:
        while rclpy.ok():
            try:
                text = self.transcribe_once()
                if not text:
                    continue

                command = self.detect_command(text)
                if command is None:
                    self.get_logger().info("No movement command detected.")
                    continue

                self.execute_command(command)
                time.sleep(self.command_hold_seconds)

            except sr.WaitTimeoutError:
                self.get_logger().debug("Listening timeout; retrying.")
            except KeyboardInterrupt:
                self.get_logger().info("Shutting down...")
                break
            except Exception as exc:
                self.get_logger().error(f"Error: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VoiceControllerNode()

    try:
        node.run()
    finally:
        node.publish_velocity(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
