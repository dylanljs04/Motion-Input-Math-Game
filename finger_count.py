import cv2
import mediapipe as mp
import random
import time

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils  # For drawing hand landmarks

# Initialize OpenCV for webcam feed
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Function to count fingers
def count_fingers(landmarks):
    # Index of tips of each finger
    finger_tips = [4, 8, 12, 16, 20]
    count = 0

    # Thumb
    if landmarks[finger_tips[0]].x < landmarks[finger_tips[0] - 2].x:
        count += 1

    # Other four fingers
    for i in range(1, 5):
        if landmarks[finger_tips[i]].y < landmarks[finger_tips[i] - 2].y:
            count += 1

    return count

# Function to generate a math question
def get_math_question():
    num1 = random.randint(1, 5)
    num2 = random.randint(1, 5)
    question = f"{num1} + {num2} = ?"
    answer = num1 + num2
    return question, answer

def main():
    # Generate a math question once, before the loop
    question, answer = get_math_question()
    
    # Timer variables
    correct_start_time = None  # To track the time the correct answer is held
    correct_display_start_time = None  # To track how long "Correct!" is displayed

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        total_fingers = 0  # Initialize total finger count

        # Draw the hand annotations on the image.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Count fingers for the current hand
                finger_count = count_fingers(hand_landmarks.landmark)
                total_fingers += finger_count  # Add to total finger count

        # Display the total finger count
        cv2.putText(image, f'Total Fingers: {total_fingers}', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the math question
        cv2.putText(image, question, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)

        # Check if the answer is correct and if the correct answer has been held for 3 seconds
        if total_fingers == answer:
            if correct_start_time is None:
                correct_start_time = time.time()  # Start the timer for holding the correct answer

            # Check if 3 seconds have passed with the correct answer held
            elapsed_time = time.time() - correct_start_time
            if elapsed_time >= 3:
                # Start "Correct!" display timer if it hasn't been started yet
                if correct_display_start_time is None:
                    correct_display_start_time = time.time()

                # Show "Correct!" message
                cv2.putText(image, "Correct!", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Check if "Correct!" has been displayed for 3 seconds
                if time.time() - correct_display_start_time >= 1.5:
                    question, answer = get_math_question()  # Generate a new question
                    correct_start_time = None  # Reset the holding timer
                    correct_display_start_time = None  # Reset the display timer
        else:
            # Reset timers if the answer is incorrect
            correct_start_time = None
            correct_display_start_time = None
            # cv2.putText(image, "Try Again!", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

        # Show the image with OpenCV
        cv2.imshow('Finger Counter', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
