import cv2
import mediapipe as mp
from pynput.mouse import Controller, Button

# Initialize MediaPipe Hand and pynput Mouse
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
mouse = Controller()

# Function to calculate distance
def calculate_distance(point1, point2):
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # Convert image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get key landmarks
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            palm_center = hand_landmarks.landmark[9]

            # Detect Pinch Gesture for Left Click
            pinch_distance = calculate_distance(thumb_tip, index_tip)
            if pinch_distance < 0.05:  # Adjust threshold based on your setup
                mouse.click(Button.left, 1)

            # Detect Two-Finger Pinch for Right Click
            scroll_distance = calculate_distance(index_tip, middle_tip)
            if pinch_distance < 0.05 and scroll_distance < 0.1:
                mouse.click(Button.right, 1)

            # Detect Hand Movement for Mouse Movement
            screen_width, screen_height = 1920, 1080  # Adjust to your screen resolution
            x, y = int(index_tip.x * screen_width), int(index_tip.y * screen_height)
            mouse.position = (x, y)

    # Display the camera feed
    cv2.imshow("Hand Gesture Mouse", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
