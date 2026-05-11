import cv2
import numpy as np


def detect_round_objects(input_path, output_path):
    dp = 1.2
    min_dist = 25
    param1 = 80
    param2 = 20
    min_radius = 5
    max_radius = 25

    image = cv2.imread(input_path)
    
    #scale = 0.25
    #small = cv2.resize(image, None, fx=scale, fy=scale)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=param1, 
                                param2=param2, minRadius=min_radius, maxRadius=max_radius)

    detected_count = 0

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)

            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            detected_count += 1

    cv2.imwrite(output_path, image)

    print(f"Detected {detected_count} round objects.")
    print(f"Saved output to: {output_path}")
