import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


model_path = r'D:\Itransition_tasks\Extra_solution\Digits_traker\mnist_model.h5'
digits_folder = r'D:\Itransition_tasks\Extra_solution\Digits_traker\digits\digits'


if not os.path.exists(model_path):
    print(f"Model file '{model_path}' not found. Please place it in the same directory as this script.")
    exit()


model = load_model(model_path)
digit_count = [0] * 10


if not os.path.exists(digits_folder):
    print(f"Digits folder '{digits_folder}' not found.")
    exit()


for file_name in os.listdir(digits_folder):
    file_path = os.path.join(digits_folder, file_name)
       
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Unable to load image {file_name}. Skipping.")
        continue
        
    img_resized = cv2.resize(img, (28, 28))       
    img_resized = img_resized / 255.0    
    img_resized = np.reshape(img_resized, (1, 28, 28, 1))
    
    prediction = model.predict(img_resized)
    recognized_digit = np.argmax(prediction)  
    digit_count[recognized_digit] += 1


print("Digit counts: ", digit_count)
