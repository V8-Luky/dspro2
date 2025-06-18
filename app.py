import cv2 
import torch
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
from models.asl_improved import ASLClassifierImproved
import time
import sys
from collections import deque, Counter

try:
    # Load model from .ckpt (state_dict has nested key prefix 'model.model.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ASLClassifierImproved().to(device)
    checkpoint = torch.load("models/asl_improved.ckpt", map_location=device)

    # Adjust state_dict keys by removing 'model.model.' prefix
    state_dict = {k.replace("model.model.", "model."): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Load label encoder
    y_test = np.load("labels/y_test.npy")
    label_encoder = LabelEncoder()
    label_encoder.fit(y_test)

    # MediaPipe setup
    mp_hands = mp.solutions.hands
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

    # Settings
    center_box = (0.3, 0.7)
    fps = 20
    last_prediction = None
    last_capture_time = 0
    capture_interval = 0.05
    prediction_history = deque(maxlen=10)

    mode = "Letter"
    output_string = ""
    last_added_letter = ""
    same_letter_cooldown = 0
    last_history_pop_time = time.time()
    last_hand_visible_time = time.time()
    history_popping_active = False
    hand_visible = False

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Failed to access the webcam. Try a different camera index or backend.")

    cv2.namedWindow("ASL Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ASL Demo", 1280, 960)
    print("Hold your hand in the center box to trigger prediction every 0.05 seconds... (Press ESC or 'q' to quit, F to switch mode, BACKSPACE to delete)")

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame. Exiting...")
                break

            h, w, _ = frame.shape
            box_width = box_height = int(min(w, h) * (center_box[1] - center_box[0]))
            cx1, cx2 = (w - box_width) // 2, (w + box_width) // 2
            cy1, cy2 = (h - box_height) // 2, (h + box_height) // 2

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            hand_visible = False

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                if len(hand.landmark) == 21:
                    landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand.landmark])

                    cx_pixel, cy_pixel = int(np.mean(landmarks[:, 0]) * w), int(np.mean(landmarks[:, 1]) * h)
                    if cx1 < cx_pixel < cx2 and cy1 < cy_pixel < cy2:
                        hand_visible = True
                        last_hand_visible_time = time.time()

                        overlay = frame.copy()
                        mp.solutions.drawing_utils.draw_landmarks(
                            overlay, hand, mp_hands.HAND_CONNECTIONS, drawing_spec, drawing_spec
                        )
                        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

                        if time.time() - last_capture_time >= capture_interval:
                            input_tensor = torch.tensor(landmarks.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
                            with torch.no_grad():
                                output = model(input_tensor)
                                pred_idx = torch.argmax(output, dim=1).item()
                                pred_label = label_encoder.inverse_transform([pred_idx])[0]
                                last_prediction = pred_label
                                prediction_history.append(last_prediction)
                                last_capture_time = time.time()

                                print(f"Capturing and predicting... Current prediction: {last_prediction}")

            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 255), 2)

            if len(prediction_history) == 10:
                most_common_pred = Counter(prediction_history).most_common(1)[0][0]
                cv2.putText(
                    frame,
                    f"Most Common: {most_common_pred}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 0, 0),
                    3
                )

                if hand_visible and mode == "Word":
                    if most_common_pred != last_added_letter:
                        output_string += most_common_pred
                        last_added_letter = most_common_pred
                        same_letter_cooldown = 5
                    elif same_letter_cooldown > 0:
                        same_letter_cooldown -= 1
                    else:
                        output_string += most_common_pred
                        same_letter_cooldown = 5

                if len(prediction_history) >= 10:
                    history_popping_active = True

                if history_popping_active and time.time() - last_history_pop_time >= 2 * capture_interval:
                    if prediction_history:
                        popped = prediction_history.popleft()
                        print(f"Popped from prediction history: {popped}")
                        last_history_pop_time = time.time()
                    if len(prediction_history) == 0:
                        history_popping_active = False
                        print("Prediction history cleared.")

            if not hand_visible and time.time() - last_hand_visible_time >= 2.0:
                if prediction_history:
                    prediction_history.clear()
                    history_popping_active = False
                    print("No hand detected for 2 seconds. Prediction history cleared.")

                # Draw "No hand detected" in red in center of box
                cv2.putText(
                    frame,
                    "No hand detected",
                    (cx1 + box_width // 4, cy1 + box_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3
                )

            cv2.putText(
                frame,
                f"Mode: {mode} (F to switch)",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

            if mode == "Word":
                cv2.putText(
                    frame,
                    f"Word: {output_string}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )

            cv2.putText(
                frame,
                "Press ESC or Q to exit",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1
            )

            cv2.imshow("ASL Demo", frame)

            key = cv2.waitKey(int(1000 / fps)) & 0xFF
            if key == ord('q') or key == 27:
                print("Exit command received. Shutting down...")
                break
            elif key == ord('f'):
                mode = "Word" if mode == "Letter" else "Letter"
                print(f"Mode switched to: {mode}")
                if mode == "Letter":
                    output_string = ""
                    last_added_letter = ""
                    same_letter_cooldown = 0
                    history_popping_active = False
                    last_history_pop_time = time.time()
                    last_hand_visible_time = time.time()
            elif key == 8:
                if output_string:
                    output_string = output_string[:-1]
                    print(f"Deleted last letter, current word: {output_string}")

except KeyboardInterrupt:
    print("KeyboardInterrupt received. Exiting...")

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    if 'cap' in locals() and cap.isOpened():
        cap.release()
        print("Webcam released.")
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print("All OpenCV windows closed.")
