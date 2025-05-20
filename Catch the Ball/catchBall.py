import cv2
import mediapipe as mp
import pygame
import sys
import random

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 640, 480
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Catch the Ball with Finger")

# Game Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BALL_COLOR = (255, 0, 0)
PADDLE_COLOR = (0, 255, 0)

# Paddle
paddle_width = 100
paddle_height = 10
paddle_y = HEIGHT - 40

# Ball
ball_radius = 15
ball_x = random.randint(0, WIDTH)
ball_y = 0
ball_speed = 5

# Score
score = 0
font = pygame.font.SysFont(None, 36)

# Camera
cap = cv2.VideoCapture(0)

clock = pygame.time.Clock()

def draw_window(paddle_x):
    global ball_x, ball_y, score

    win.fill(BLACK)
    
    # Draw paddle
    pygame.draw.rect(win, PADDLE_COLOR, (paddle_x, paddle_y, paddle_width, paddle_height))

    # Draw ball
    pygame.draw.circle(win, BALL_COLOR, (ball_x, ball_y), ball_radius)

    # Ball movement
    ball_y += ball_speed

    # Collision detection
    if paddle_y < ball_y + ball_radius < paddle_y + paddle_height:
        if paddle_x < ball_x < paddle_x + paddle_width:
            score += 1
            ball_x = random.randint(0, WIDTH)
            ball_y = 0

    if ball_y > HEIGHT:
        score -= 1
        ball_x = random.randint(0, WIDTH)
        ball_y = 0

    # Show score
    score_text = font.render(f"Score: {score}", True, WHITE)
    win.blit(score_text, (10, 10))

    pygame.display.update()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror view

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    finger_x = WIDTH // 2  # Default center position

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip position (landmark 8)
            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[8].x * WIDTH)
            finger_x = x - paddle_width // 2

    # Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit()

    draw_window(finger_x)
    clock.tick(60)

    # Show webcam in a small OpenCV window (optional)
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
