import cv2
import mediapipe as mp
import pyautogui
import math
import time
import os

# ============================================================
# SETTINGS
# ============================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

SMOOTHING = 5
CLICK_THRESHOLD = 30
CLICK_COOLDOWN = 0.5

# ============================================================
# PYAutoGUI
# ============================================================

pyautogui.PAUSE = 0.01
pyautogui.MINIMUM_DURATION = 0

screen_width, screen_height = pyautogui.size()

# ============================================================
# MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task"
)

if not os.path.isfile(MODEL_PATH):
    print("ERROR: hand_landmarker.task not found!")
    print("Expected location:")
    print(MODEL_PATH)
    input("Press Enter to exit...")
    raise SystemExit

print("Model found:")
print(MODEL_PATH)

# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# FUNCTIONS
# ============================================================

def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def get_landmarks(hand, width, height):

    points = []

    for landmark in hand:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    return points


def fingers_up(points):

    index = points[8][1] < points[6][1]

    middle = points[12][1] < points[10][1]

    ring = points[16][1] < points[14][1]

    pinky = points[20][1] < points[18][1]

    return index, middle, ring, pinky


# ============================================================
# CAMERA
# ============================================================

print("Opening webcam...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print()
    print("ERROR: Could not open webcam.")
    print()
    print("Try changing:")
    print("camera = cv2.VideoCapture(0)")
    print("to:")
    print("camera = cv2.VideoCapture(1)")
    print()

    input("Press Enter to exit...")
    raise SystemExit

print("Webcam opened successfully.")

# ============================================================
# VARIABLES
# ============================================================

previous_x = screen_width // 2
previous_y = screen_height // 2

last_left_click = 0
last_right_click = 0

# ============================================================
# RUN MEDIAPIPE
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    print()
    print("=" * 50)
    print("AI VIRTUAL MOUSE RUNNING")
    print("=" * 50)
    print()
    print("INDEX FINGER  = Move")
    print("PINCH          = Left Click")
    print("TWO FINGERS    = Right Click")
    print("OPEN HAND      = Scroll")
    print("Q              = Quit")
    print()

    while True:

        # ----------------------------------------------------
        # Read webcam
        # ----------------------------------------------------

        success, frame = camera.read()

        if not success:
            print("Failed to read webcam.")
            continue

        # Mirror image
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # ----------------------------------------------------
        # Convert image
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # MediaPipe image
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # ----------------------------------------------------
        # Detect hand
        # ----------------------------------------------------

        result = landmarker.detect(mp_image)

        gesture = "NO HAND"

        # ====================================================
        # HAND FOUND
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            points = get_landmarks(
                hand,
                width,
                height
            )

            # ------------------------------------------------
            # Draw points
            # ------------------------------------------------

            for point in points:

                cv2.circle(
                    frame,
                    point,
                    4,
                    (0, 255, 0),
                    -1
                )

            # ------------------------------------------------
            # Draw hand connections
            # ------------------------------------------------

            connections = [

                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),

                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),

                (5, 9),
                (9, 10),
                (10, 11),
                (11, 12),

                (9, 13),
                (13, 14),
                (14, 15),
                (15, 16),

                (13, 17),
                (17, 18),
                (18, 19),
                (19, 20),

                (0, 17)
            ]

            for start, end in connections:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 0, 0),
                    2
                )

            # ------------------------------------------------
            # Fingers
            # ------------------------------------------------

            (
                index_up,
                middle_up,
                ring_up,
                pinky_up
            ) = fingers_up(points)

            # ------------------------------------------------
            # Pinch
            # ------------------------------------------------

            pinch = distance(
                points[4],
                points[8]
            )

            # =================================================
            # LEFT CLICK
            # =================================================

            if pinch < CLICK_THRESHOLD:

                gesture = "LEFT CLICK"

                now = time.time()

                if now - last_left_click > CLICK_COOLDOWN:

                    pyautogui.click(
                        _pause=False
                    )

                    last_left_click = now

            # =================================================
            # RIGHT CLICK
            # =================================================

            elif (
                index_up
                and middle_up
                and not ring_up
                and not pinky_up
            ):

                gesture = "RIGHT CLICK"

                now = time.time()

                if now - last_right_click > CLICK_COOLDOWN:

                    pyautogui.rightClick(
                        _pause=False
                    )

                    last_right_click = now

            # =================================================
            # MOVE
            # =================================================

            elif (
                index_up
                and not middle_up
                and not ring_up
                and not pinky_up
            ):

                gesture = "MOVE"

                index_x = points[8][0]
                index_y = points[8][1]

                target_x = int(
                    index_x / width * screen_width
                )

                target_y = int(
                    index_y / height * screen_height
                )

                current_x = (
                    previous_x +
                    (target_x - previous_x)
                    / SMOOTHING
                )

                current_y = (
                    previous_y +
                    (target_y - previous_y)
                    / SMOOTHING
                )

                pyautogui.moveTo(
                    int(current_x),
                    int(current_y),
                    _pause=False
                )

                previous_x = current_x
                previous_y = current_y

            # =================================================
            # OPEN HAND / SCROLL
            # =================================================

            elif (
                index_up
                and middle_up
                and ring_up
                and pinky_up
            ):

                gesture = "SCROLL"

                index_y = points[8][1]

                if index_y < height * 0.30:

                    pyautogui.scroll(
                        2,
                        _pause=False
                    )

                elif index_y > height * 0.70:

                    pyautogui.scroll(
                        -2,
                        _pause=False
                    )

            else:

                gesture = "PAUSE"

        # ====================================================
        # UI
        # ====================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (300, 55),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            "Gesture: " + gesture,
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Q = Quit",
            (10, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "AI Virtual Mouse",
            frame
        )

        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print()
print("Virtual Mouse stopped successfully.")