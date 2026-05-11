import cv2
import numpy as np


def detect_round_objectsC(input_path, output_path):
    dp = 1.2
    min_dist = 25
    param1 = 80
    param2 = 20
    min_radius = 5
    max_radius = 25
    light_threshold = 100
    required_light_ratio = 0.40
    image = cv2.imread(input_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")

    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=param1, 
                                param2=param2, minRadius=min_radius, maxRadius=max_radius)

    kept_count = 0
    removed_count = 0

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for (cx, cy, r) in circles:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            pixels_inside = gray[mask == 255]
            if len(pixels_inside) == 0:
                continue

            light_pixels = pixels_inside > light_threshold
            light_ratio = np.sum(light_pixels) / len(pixels_inside)
            if light_ratio >= required_light_ratio:
                kept_count += 1
                cv2.circle(output, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(output, (cx, cy), 2, (0, 0, 255), -1)

            else:
                removed_count += 1
                cv2.circle(output, (cx, cy), r, (0, 0, 255), 1)

    cv2.imwrite(output_path, output)

    print(f"Kept circles: {kept_count}")
    print(f"Removed circles: {removed_count}")
    print(f"Saved output to: {output_path}")
