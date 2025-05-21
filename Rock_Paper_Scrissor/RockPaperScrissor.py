import cv2
import mediapipe as mp
import random
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Game choices
choices = ['Rock', 'Paper', 'Scissors']

# Helper function to classify gesture
def classify_gesture(hand_landmarks):
    fingers = []

    # Thumb
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < \
       hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Fingers (Index to Pinky)
    tip_ids = [8, 12, 16, 20]
    for tip in tip_ids:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    total_fingers = fingers.count(1)

    if total_fingers == 0:
        return 'Rock'
    elif total_fingers == 2 and fingers[1] and fingers[2]:  # index and middle
        return 'Scissors'
    elif total_fingers >= 4:
        return 'Paper'
    else:
        return 'Unknown'

# Game logic
def get_winner(player, ai):
    if player == ai:
        return "Draw"
    if (player == "Rock" and ai == "Scissors") or \
       (player == "Scissors" and ai == "Paper") or \
       (player == "Paper" and ai == "Rock"):
        return "You Win!"
    return "AI Wins!"

# Start webcam
cap = cv2.VideoCapture(0)
start_time = time.time()
ai_move = random.choice(choices)
cooldown = 3
result = ""

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    player_move = "Waiting..."

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            player_move = classify_gesture(hand_landmarks)

    # Update every 3 seconds
    if time.time() - start_time > cooldown:
        if player_move in choices:
            ai_move = random.choice(choices)
            result = get_winner(player_move, ai_move)
        start_time = time.time()

    # Display UI
    cv2.rectangle(frame, (0, 0), (640, 100), (0, 0, 0), -1)
    cv2.putText(frame, f"Your Move: {player_move}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"AI Move: {ai_move}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"Result: {result}", (350, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Rock-Paper-Scissors", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
