import cv2
import numpy as np
import os
from tqdm import tqdm

# XDOG filter parameters
Gamma = 0.98
Phi = 200
Epsilon = -0.1
k = 1.6
Sigma = 0.8

def apply_xdog_filter(inputIm):
    inputIm = cv2.cvtColor(inputIm, cv2.COLOR_BGR2GRAY)
    inputIm = inputIm.astype(np.float64) / 255.0

    # Gauss Filters
    gFilteredIm1 = cv2.GaussianBlur(inputIm, (0, 0), Sigma)
    gFilteredIm2 = cv2.GaussianBlur(inputIm, (0, 0), Sigma * k)

    differencedIm2 = gFilteredIm1 - (Gamma * gFilteredIm2)

    x, y = differencedIm2.shape

    # Extended difference of gaussians
    for i in range(x):
        for j in range(y):
            if differencedIm2[i, j] < Epsilon:
                differencedIm2[i, j] = 1
            else:
                differencedIm2[i, j] = 1 + np.tanh(Phi * (differencedIm2[i, j]))

    # Normalize to range [0, 1]
    xdog_filtered_image = cv2.normalize(differencedIm2, None, 0, 1, cv2.NORM_MINMAX)
    
    # Thresholding
    meanValue = np.mean(xdog_filtered_image)
    xdog_filtered_image[xdog_filtered_image <= meanValue] = 0.0
    xdog_filtered_image[xdog_filtered_image > meanValue] = 1.0
    
    return (xdog_filtered_image * 255).astype(np.uint8)

def process_images(input_dir, output_dir):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each image in the input directory
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Read image
            inputIm = cv2.imread(input_path)
            if inputIm is None:
                print(f"Failed to load image {input_path}")
                continue
            
            # Apply XDOG filter
            xdog_filtered_image = apply_xdog_filter(inputIm)
            
            # Save the result
            cv2.imwrite(output_path, xdog_filtered_image)
            # print(f"Processed {input_path} -> {output_path}")

# Define input and output directories
input_dirs = [
    "/content/drive/MyDrive/Physics Wallah Internship/Dataset/Train/Input",
    "/content/drive/MyDrive/Physics Wallah Internship/Dataset/Test/Input",
    "/content/drive/MyDrive/Physics Wallah Internship/Dataset/Val/Input"
]

output_dirs = [
    "/content/drive/MyDrive/Physics Wallah Internship/Dataset/Train/Output",
    "/content/drive/MyDrive/Physics Wallah Internship/Dataset/Test/Output",
    "/content/drive/MyDrive/Physics Wallah Internship/Dataset/Val/Output"
]

# Process the images for each input/output directory pair
for input_dir, output_dir in zip(input_dirs, output_dirs):
    process_images(input_dir, output_dir)
