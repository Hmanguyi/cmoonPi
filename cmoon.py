import cv2
import numpy as np
import base64
import os
import time
from mistralai import Mistral
import datetime
import threading
from typing import List, Tuple
import RPi.GPIO as GPIO
from openai import OpenAI

fail_count = 0
FAIL_LIMIT = 10
startup_frames = 5
frame_count = 0
the_class_des = None

def safe_shutdown(reason="Camera disconnected"):
    print(f"⚠️ SHUTDOWN: {reason}")

    # Stop motors here
    motor_stop_a()
    motor_stop_b()

    # Turn off LED
    GPIO.output(LED_PIN, GPIO.LOW)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()

    time.sleep(1)

    # Shutdown system
    os.system("sudo shutdown now")

currentObject = None
startTime = None

motion_pause_until = 0

captured_image = None
captured_image_expires_at = 0

ledOn = 0

GPIO.setmode(GPIO.BOARD)

IN1, IN2, ENA = 38, 36, 40
IN3, IN4, ENB = 16, 18, 22

LED_PIN = 35  # GPIO19

GPIO.setup(LED_PIN, GPIO.OUT)

GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)

pwma = GPIO.PWM(ENA, 1000)
pwmb = GPIO.PWM(ENB, 1000)
pwma.start(0)
pwmb.start(0)

GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT)

def led_on():
    GPIO.output(LED_PIN, GPIO.HIGH)

def led_off():
    GPIO.output(LED_PIN, GPIO.LOW)

# ---------------- Motor Functions ----------------
def motor_forward_a(speed=100):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    pwma.ChangeDutyCycle(speed)

def motor_backward_a(speed=100):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    pwma.ChangeDutyCycle(speed)

def motor_stop_a():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.HIGH)
    pwma.ChangeDutyCycle(0)

def motor_forward_b(speed=100):
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmb.ChangeDutyCycle(speed)

def motor_backward_b(speed=100):
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwmb.ChangeDutyCycle(speed)

def motor_stop_b():
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.HIGH)
    pwmb.ChangeDutyCycle(0)

# ---------------- Color Detection ----------------
LOWER_GREEN = np.array([25, 40, 40])
UPPER_GREEN = np.array([95, 255, 255])
roi_coords = (48, 294, 39, 100)
fill_threshold = 0.80

LOWER_BLUE = np.array([90, 50, 50])
UPPER_BLUE = np.array([130, 255, 255])

LOWER_RED_1 = np.array([0, 50, 50])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 50, 50])
UPPER_RED_2 = np.array([179, 255, 255])

def green_detected(frame):
    x, y, w, h = roi_coords
    roi = frame[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    fill_ratio = cv2.countNonZero(mask) / (w * h)
    return fill_ratio >= fill_threshold, mask

def blue_detected(frame):
    x, y, w, h = roi_coords
    roi = frame[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    fill_ratio = cv2.countNonZero(mask) / (w * h)
    return fill_ratio >= fill_threshold, mask

def red_detected(frame):
    x, y, w, h = roi_coords
    roi = frame[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.bitwise_or(mask1, mask2)
    fill_ratio = cv2.countNonZero(mask) / (w * h)
    return fill_ratio >= fill_threshold, mask

def drop():
    motor_backward_b(100)
    time.sleep(1.1)
    motor_stop_b()
    time.sleep(0.2)
    motor_forward_b(100)
    time.sleep(1.3)
    motor_stop_b()

os.environ["OPENAI_API_KEY"] = "" 
client = OpenAI()

def frame_to_base64(frame):
    _, buffer = cv2.imencode(".png", frame)
    return base64.b64encode(buffer).decode("utf-8")

def describe_image_from_frame(frame) -> str:
    img_b64 = frame_to_base64(frame)
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe the object in this image clearly and briefly."},
                {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"}
            ]
        }],
        temperature=0
    )
    return response.output[0].content[0].text.strip()

def classify_text(text: str) -> str:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=f"""
You are an expert environmental waste classification assistant.


Classify the object below into EXACTLY one word:
trash
recyclable
compostable


Rules:
- Output ONE word only
- If ambiguous, choose the safest environmental option
- Dirty food containers are trash unless clearly compostable


Object description:
{text}
""",
        temperature=0
    )
    return response.output[0].content[0].text.strip().lower()

def draw_paragraph(img, text, start=(20, 40), line_height=25):
    x, y = start
    for line in text.split("\n"):
        cv2.putText(img, line, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)

save_dir = "motion_captures"
os.makedirs(save_dir, exist_ok=True)

led_on()
time.sleep(2)
led_off()

cap = cv2.VideoCapture(0)

text_window = np.zeros((300, 600, 3), dtype=np.uint8)

ret, frame1 = cap.read()
prev_gray = cv2.GaussianBlur(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), (21, 21), 0)

try:
    while True:
        ret, frame2 = cap.read()
        if not ret:
            safe_shutdown()
            break

        detected, mask = green_detected(frame2)
        x, y, w, h = roi_coords
        color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 2)

        gray = cv2.GaussianBlur(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        current_time = time.time()

        if current_time >= motion_pause_until:
            # Motion detection is active
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=2)

            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False
            frame_area = frame2.shape[0] * frame2.shape[1]

            for contour in contours:
                if cv2.contourArea(contour) > 230400 and cv2.contourArea(contour) < 230400:
                    continue
                x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
                cv2.rectangle(frame2, (x_c, y_c), (x_c + w_c, y_c + h_c), (0, 0, 0), 2)
                motion_detected = True
                if startTime is None:
                    startTime = time.time()

            # If motion detected, save frame
            if motion_detected and (time.time() - startTime) >= 2:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                captured_image = frame2.copy()
                captured_image_expires_at = current_time + 3
                motion_pause_until = current_time + 4

                led_on()
                time.sleep(0.5)
                led_off()
                led_on()
                time.sleep(0.5)
                led_off()

                text = describe_image_from_frame(captured_image)
                text = classify_text(text)
                currentObject = text
                the_class_des = text

                cv2.putText(captured_image, text, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("class", captured_image)
                text_window[:] = 0
                draw_paragraph(text_window, the_class_des)
                cv2.imshow("Classification Text", text_window)

                startTime = None

        # Process detected objects
        if currentObject:
            obj_lower = currentObject.lower()
            if obj_lower == "compostable":
                print("Recyclable detected")
                motor_backward_a(100)

                while True:
                    ret, frame2 = cap.read()
                    if not ret:
                        safe_shutdown()
                        break
                    frame_count += 1
                    x, y, w, h = roi_coords
                    if frame_count <= startup_frames:
                        color = (255, 255, 0)
                        cv2.putText(frame2, f"Waiting... {frame_count}/{startup_frames}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        detected = False
                    else:
                        detected, mask = blue_detected(frame2)
                        color = (0, 255, 0) if detected else (0, 0, 255)

                    cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 2)
                    cv2.imshow("Motion Detection", frame2)
                    cv2.waitKey(1)
                    if detected:
                        motor_stop_a()
                        print("✅ Green detected (after startup delay), motor stopped")
                        break

                drop()
                time.sleep(1)
                motor_forward_a(100)
                startup_frames = 5
                frame_count = 0

                while True:
                    ret, frame2 = cap.read()
                    if not ret:
                        safe_shutdown()
                        break
                    frame_count += 1
                    x, y, w, h = roi_coords
                    if frame_count <= startup_frames:
                        color = (255, 255, 0)
                        cv2.putText(frame2, f"Waiting... {frame_count}/{startup_frames}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        detected = False
                    else:
                        detected, mask = green_detected(frame2)
                        color = (0, 255, 0) if detected else (0, 0, 255)

                    cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 2)
                    cv2.imshow("Motion Detection", frame2)
                    cv2.waitKey(1)
                    if detected:
                        print("✅ Green detected (after startup delay), motor stopped")
                        break

                startup_frames = 5
                frame_count = 0
                startTime = None
                motion_pause_until = time.time() + 1
                prev_gray = gray.copy()
                captured_image = None
                captured_image_expires_at = 0
                currentObject = None
                print("Recyclable done")

            elif obj_lower == "recyclable":
                print("Compostable detected")
                drop()
                print("Compostable done")

            elif obj_lower == "trash":
                print("Trash detected")
                motor_backward_a(100)

                while True:
                    ret, frame2 = cap.read()
                    if not ret:
                        safe_shutdown()
                        break
                    frame_count += 1
                    x, y, w, h = roi_coords
                    if frame_count <= startup_frames:
                        color = (255, 255, 0)
                        cv2.putText(frame2, f"Waiting... {frame_count}/{startup_frames}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        detected = False
                    else:
                        detected, mask = blue_detected(frame2)
                        color = (0, 255, 0) if detected else (0, 0, 255)

                    cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 2)
                    cv2.imshow("Motion Detection", frame2)
                    cv2.waitKey(1)
                    if detected:
                        motor_stop_a()
                        print("✅ Green detected (after startup delay), motor stopped")
                        break

                drop()
                time.sleep(1)
                motor_forward_a(100)
                startup_frames = 5
                frame_count = 0

                while True:
                    ret, frame2 = cap.read()
                    if not ret:
                        safe_shutdown()
                        break
                    frame_count += 1
                    x, y, w, h = roi_coords
                    if frame_count <= startup_frames:
                        color = (255, 255, 0)
                        cv2.putText(frame2, f"Waiting... {frame_count}/{startup_frames}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        detected = False
                    else:
                        detected, mask = green_detected(frame2)
                        color = (0, 255, 0) if detected else (0, 0, 255)

                    cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 2)
                    cv2.imshow("Motion Detection", frame2)
                    cv2.waitKey(1)
                    if detected:
                        motor_stop_a()
                        print("✅ Green detected (after startup delay), motor stopped")
                        break

                startup_frames = 5
                frame_count = 0
                startTime = None
                motion_pause_until = time.time() + 1
                prev_gray = gray.copy()
                captured_image = None
                captured_image_expires_at = 0
                currentObject = None
                print("Trash done")

        if current_time < motion_pause_until:
            cv2.putText(frame2, "⏸ Motion detection paused", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Motion Detection", frame2)

        if captured_image is not None and current_time < captured_image_expires_at:
            cv2.namedWindow("Captured Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Captured Image", 400, 300)
            cv2.moveWindow("Captured Image", 1000, -100)
            cv2.imshow("Captured Image", captured_image)
        elif captured_image is not None and current_time >= captured_image_expires_at:
            cv2.destroyWindow("Captured Image")
            captured_image = None

        cv2.imshow("Motion Detection", frame2)
        prev_gray = gray

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    print("🧹 Cleaning up GPIO and resources")
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
