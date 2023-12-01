import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from tensorflow.keras import layers

# Step 1: Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Step 2: Initialize variables
data = []
labels = []
class_names = ["One", "Two", "Three", "Four"]

# Step 3: Set up paths
base_folder = "data_videos"

# Step 4: Loop through each class folder
for class_name in class_names:
    class_folder = os.path.join(base_folder, class_name)

    # Step 5: Loop through each video file in the class folder
    for filename in os.listdir(class_folder):
        if filename.endswith(".mov"):
            video_path = os.path.join(class_folder, filename)

            # Step 6: Capture video from the file
            cap = cv2.VideoCapture(video_path)

            # Step 7: Collect training data
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Hand landmarks detection using MediaPipe
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0].landmark

                    
                    data.append([landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks])
                    labels.append(class_names.index(class_name))

                    # Draw hand landmarks on the frame
                    for landmark in landmarks:
                        h, w, _ = frame.shape
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                cv2.imshow("Collecting Data", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()

# Step 8: Convert data and labels to numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels, dtype="int")

# Step 9: Train a simple model using TensorFlow
model = keras.Sequential([
    layers.InputLayer(input_shape=(42,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(data, labels, epochs=50)

# Step 10: Save the trained model
model.save("hand_sign_model.h5")

# Step 11: Release resources
cv2.destroyAllWindows()
