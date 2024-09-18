import os
import cv2

# Initialize an array to count occurrences of each digit (0-9)
digit_count = [0] * 10

# Function to process each image, assuming it's a digit image
def process_image(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Use thresholding to binarize the image (make it purely black and white)
    _, thresh_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours to detect the digit
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Use the largest contour assuming it represents the digit
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        digit_img = thresh_img[y:y+h, x:x+w]

        # Resize the digit to a standard size (e.g., 28x28)
        digit_img = cv2.resize(digit_img, (28, 28))

        # Return the processed digit image for further processing (e.g., OCR)
        return digit_img

# Correct path to the folder containing digit images
digits_folder = r'D:\Itransition_tasks\Extra_solution\Digits_traker\digits\digits'

# Process all images in the 'digits' folder
for file_name in os.listdir(digits_folder):
    file_path = os.path.join(digits_folder, file_name)

    # Ensure the file name starts with a digit (0-9)
    if file_name[0].isdigit():
        recognized_digit = int(file_name[0])
        digit_count[recognized_digit] += 1

# Print the result
print("Digit counts: ", digit_count)
