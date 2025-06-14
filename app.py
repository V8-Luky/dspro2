import cv2 
import torch
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
from models.asl_improved import ASLClassifierImproved
import time
import sys

try:
    # Load model from .ckpt (state_dict has nested key prefix 'model.model.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

    # Drawing style
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

    # Settings
    center_box = (0.3, 0.7)
    fps = 20
    last_prediction = None
    last_capture_time = 0
    capture_interval = 1.0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("❌ Failed to access the webcam. Try a different camera index or backend.")

    cv2.namedWindow("ASL Demo")
    print("Hold your hand in the center box to trigger prediction every second... (Press ESC or 'q' to quit)")

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

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            if len(hand.landmark) == 21:
                landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand.landmark])

                # Convert normalized coordinates to pixels for center check
                cx_pixel, cy_pixel = int(np.mean(landmarks[:, 0]) * w), int(np.mean(landmarks[:, 1]) * h)
                if cx1 < cx_pixel < cx2 and cy1 < cy_pixel < cy2:
                    overlay = frame.copy()
                    mp.solutions.drawing_utils.draw_landmarks(overlay, hand, mp_hands.HAND_CONNECTIONS, drawing_spec, drawing_spec)
                    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

                    if time.time() - last_capture_time >= capture_interval:
                        print("Capturing and predicting...")
                        input_tensor = torch.tensor(landmarks.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
                        with torch.no_grad():
                            output = model(input_tensor)
                            pred_idx = torch.argmax(output, dim=1).item()
                            pred_label = label_encoder.inverse_transform([pred_idx])[0]
                            last_prediction = pred_label
                            last_capture_time = time.time()

        # Draw center box
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 255), 2)

        # Show last prediction
        if last_prediction is not None:
            cv2.putText(frame, f"Last Prediction: {last_prediction}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        cv2.imshow("ASL Demo", frame)

        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord('q') or key == 27:
            print("Exit command received. Shutting down...")
            break

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    if 'cap' in locals() and cap.isOpened():
        cap.release()
        print("Webcam released.")
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print("All OpenCV windows closed.")
