import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Camera setup - reduce resolution for faster processing
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # Reduced width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Reduced height

# Hand detection settings - keep complexity low for FPS boost
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=0  # Lower complexity for speed
)

# Constants
STEERING_DEADZONE = 5
MAX_STEERING_ANGLE = 60
CALIBRATION_TIME = 5

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

class KeyController:
    def __init__(self):
        self.states = {}
        self.last_key_time = {}

    def press_key(self, key):
        if not self.states.get(key, False):
            pyautogui.keyDown(key)
            self.states[key] = True

    def release_key(self, key):
        if self.states.get(key, False):
            pyautogui.keyUp(key)
            self.states[key] = False

    def tap_key(self, key):
        pyautogui.keyDown(key)
        pyautogui.keyUp(key)

    def update(self, analog_inputs):
        current_time = time.time()

        for key, magnitude in analog_inputs.items():
            self._handle_analog_key(key, magnitude, current_time)

    def _handle_analog_key(self, key, magnitude, current_time):
        # Release if below deadzone
        if magnitude < STEERING_DEADZONE:
            self.release_key(key)
            return

        # Tap or hold logic based on magnitude
        if STEERING_DEADZONE <= magnitude < 15:
            # Tap key every 0.2 seconds
            if (current_time - self.last_key_time.get(key, 0)) > 0.2:
                self.tap_key(key)
                self.last_key_time[key] = current_time
            self.release_key(key)

        elif 15 <= magnitude < 35:
            # Tap key faster every 0.1 seconds
            if (current_time - self.last_key_time.get(key, 0)) > 0.1:
                self.tap_key(key)
                self.last_key_time[key] = current_time
            self.release_key(key)

        else:
            # Hold key down continuously
            self.press_key(key)

    def release_all(self):
        for key in self.states:
            if self.states[key]:
                pyautogui.keyUp(key)
        self.states = {}

key_controller = KeyController()

def calculate_steering_angle(left_hand, right_hand, baseline_angle):
    left_point = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    right_point = right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    current_angle = math.degrees(math.atan2(
        right_point.y - left_point.y,
        right_point.x - left_point.x
    ))
    return current_angle - baseline_angle

def is_driving_position(left_hand, right_hand):
    if not left_hand or not right_hand:
        return False
    lw, rw = left_hand.landmark[mp_hands.HandLandmark.WRIST], right_hand.landmark[mp_hands.HandLandmark.WRIST]
    vertical_diff = abs(lw.y - rw.y)
    horizontal_dist = abs(lw.x - rw.x)
    total_dist = math.sqrt((lw.x - rw.x) ** 2 + (lw.y - rw.y) ** 2)
    return vertical_diff < 0.45 and horizontal_dist > 0.12 and total_dist > 0.15

try:
    calibration_start_time = None
    baseline_angle = 0
    calibrated = False
    prev_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        left_hand = right_hand = None
        steering_angle = 0
        driving = False

        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_type = handedness.classification[0].label
                if hand_type == "Left":
                    left_hand = hand_landmarks
                elif hand_type == "Right":
                    right_hand = hand_landmarks

        if left_hand:
            mp_drawing.draw_landmarks(frame, left_hand, mp_hands.HAND_CONNECTIONS)
        if right_hand:
            mp_drawing.draw_landmarks(frame, right_hand, mp_hands.HAND_CONNECTIONS)

        if not calibrated:
            if left_hand and right_hand and is_driving_position(left_hand, right_hand):
                if calibration_start_time is None:
                    calibration_start_time = time.time()

                elapsed = time.time() - calibration_start_time
                remaining = max(0, CALIBRATION_TIME - elapsed)

                cv2.putText(frame, f"Hold STEERING WHEEL position", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Calibrating... {remaining:.1f}s", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if remaining == 0:
                    baseline_angle = math.degrees(math.atan2(
                        right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y -
                        left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                        right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x -
                        left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
                    ))
                    calibrated = True
            else:
                calibration_start_time = None
        else:
            if left_hand and right_hand and is_driving_position(left_hand, right_hand):
                driving = True
                steering_angle = calculate_steering_angle(left_hand, right_hand, baseline_angle)
                steering_angle = max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, steering_angle))

                # Now convert steering angle to key presses for left (a) and right (d)
                # If steering_angle < -STEERING_DEADZONE -> press 'a' with magnitude proportional to abs(angle)
                # If steering_angle > STEERING_DEADZONE -> press 'd'
                analog_inputs = {}
                if steering_angle < -STEERING_DEADZONE:
                    analog_inputs['a'] = abs(steering_angle)
                    analog_inputs['d'] = 0
                elif steering_angle > STEERING_DEADZONE:
                    analog_inputs['d'] = abs(steering_angle)
                    analog_inputs['a'] = 0
                else:
                    analog_inputs['a'] = 0
                    analog_inputs['d'] = 0

                # Also press 'w' key always when driving to move forward
                analog_inputs['w'] = 100 if driving else 0

                key_controller.update(analog_inputs)

                # Draw steering line
                p1 = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                p2 = right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                p1 = (int(p1.x * frame.shape[1]), int(p1.y * frame.shape[0]))
                p2 = (int(p2.x * frame.shape[1]), int(p2.y * frame.shape[0]))
                cv2.line(frame, p1, p2, (0, 255, 0), 2)

                # Display angle info
                percent = (steering_angle / MAX_STEERING_ANGLE) * 100
                direction = "LEFT" if percent < 0 else "RIGHT" if percent > 0 else "CENTER"
                cv2.putText(frame, f"Steering: {direction} {abs(percent):.0f}%", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Steering meter
                meter_x, meter_y, meter_width = 50, frame.shape[0] - 50, 150
                cv2.rectangle(frame, (meter_x, meter_y - 15), (meter_x + meter_width, meter_y + 15), (50, 50, 50), -1)
                center_x = meter_x + meter_width // 2
                indicator_pos = center_x + int((meter_width // 2) * (percent / 100))
                cv2.rectangle(frame, (indicator_pos - 4, meter_y - 20), (indicator_pos + 4, meter_y + 20), (0, 255, 0), -1)
                cv2.line(frame, (center_x, meter_y - 20), (center_x, meter_y + 20), (255, 255, 255), 1)

            else:
                # No driving position detected, release all keys
                key_controller.release_all()

            # Display control status
            controls = "W" if key_controller.states.get('w', False) else ""
            if key_controller.states.get('a', False):
                controls += "+A(Left)"
            elif key_controller.states.get('d', False):
                controls += "+D(Right)"
            cv2.putText(frame, controls, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # FPS counter
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

        cv2.imshow("Virtual Steering Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    key_controller.release_all()
    cap.release()
    cv2.destroyAllWindows()
