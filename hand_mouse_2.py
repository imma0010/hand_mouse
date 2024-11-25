import cv2
import mediapipe as mp
from pynput.mouse import Controller, Button
import time

# Initialize MediaPipe Hand and pynput Mouse
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
mouse = Controller()

# Function to calculate distance
def calculate_distance(point1, point2):
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

# Smoothing variables
position_buffer = []
buffer_size = 5

# Dead zone variables
dead_zone = 10  # Pixels
prev_x, prev_y = 0, 0

# Timer variables
last_update_time = 0
update_interval = 0.05  # 50ms

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

            # Add to position buffer
            position_buffer.append((x, y))
            if len(position_buffer) > buffer_size:
                position_buffer.pop(0)

            # Calculate the average position
            avg_x = sum(pos[0] for pos in position_buffer) // len(position_buffer)
            avg_y = sum(pos[1] for pos in position_buffer) // len(position_buffer)

            # Apply dead zone
            if abs(avg_x - prev_x) > dead_zone or abs(avg_y - prev_y) > dead_zone:
                # Update mouse position at a fixed interval
                current_time = time.time()
                if current_time - last_update_time > update_interval:
                    mouse.position = (avg_x, avg_y)
                    prev_x, prev_y = avg_x, avg_y
                    last_update_time = current_time

    # Display the camera feed
    cv2.imshow("Hand Gesture Mouse", img)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
