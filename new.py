#!/usr/bin/env python3
import speech_recognition as sr
import roslibpy

DUCKIE_HOST = 'entebot208.local'
DUCKIE_PORT = 9090
TOPIC_NAME = '/lane_controller_node/car_cmd'
TOPIC_TYPE = 'duckietown_msgs/Twist2DStamped'

FORWARD_V = 0.4
TURN_V = 0.2
TURN_OMEGA = 1.5

client = roslibpy.Ros(host=DUCKIE_HOST, port=DUCKIE_PORT)
talker = roslibpy.Topic(client, TOPIC_NAME, TOPIC_TYPE)


def send_move(v: float, omega: float) -> None:
    if not client.is_connected:
        print("ROS bridge is not connected.")
        return

    msg = roslibpy.Message({
        'v': float(v),
        'omega': float(omega)
    })
    talker.publish(msg)
    print(f">> Command sent: v={v}, omega={omega}")


def listen_and_command() -> None:
    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=0)

    with mic as source:
        print("Voice control ready.")
        print("Say: forward, stop, left, right, exit")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        while True:
            try:
                print("\nListening...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=2)
                text = recognizer.recognize_google(audio).lower().strip()
                print(f"Heard: {text}")

                if "forward" in text:
                    send_move(FORWARD_V, 0.0)
                elif "stop" in text:
                    send_move(0.0, 0.0)
                elif "left" in text:
                    send_move(TURN_V, TURN_OMEGA)
                elif "right" in text:
                    send_move(TURN_V, -TURN_OMEGA)
                elif "exit" in text:
                    print("Exiting...")
                    break
                else:
                    print("Unknown command")

            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.WaitTimeoutError:
                print("Listening timed out")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


def main() -> None:
    print(f"Connecting to ROS bridge at ws://{DUCKIE_HOST}:{DUCKIE_PORT} ...")
    client.run(timeout=10)

    if not client.is_connected:
        print("Failed to connect to ROS bridge.")
        print("Check that:")
        print("1. entebot208.local resolves on this machine")
        print("2. rosbridge is running on port 9090")
        print("3. the topic name and message type are correct")
        return

    print("Connected to ROS bridge.")

    try:
        listen_and_command()
    finally:
        try:
            send_move(0.0, 0.0)
        except Exception:
            pass
        try:
            talker.unadvertise()
        except Exception:
            pass
        client.terminate()
        print("Stopped safely.")


if __name__ == "__main__":
    main()
