import cv2
import numpy as np


def detect_round_objectsL(input_path, output_path):
    dp = 1.2
    min_dist = 25
    param1 = 80
    param2 = 20
    min_radius = 5
    max_radius = 25
    image = cv2.imread(input_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")

    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    circle_image = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(circle_image, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=param1, 
                                param2=param2, minRadius=min_radius, maxRadius=max_radius)

    detected_count = 0
    line_count = 0

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for (cx, cy, r) in circles:
            detected_count += 1
            cv2.circle(output, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(output, (cx, cy), 2, (0, 0, 255), -1)

            x1 = max(cx - r, 0)
            y1 = max(cy - r, 0)
            x2 = min(cx + r, gray.shape[1])
            y2 = min(cy + r, gray.shape[0])
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            roi = clahe.apply(roi)

            mask = np.zeros_like(roi)
            rr = min(mask.shape[0], mask.shape[1]) // 2
            cv2.circle(mask, (mask.shape[1] // 2, mask.shape[0] // 2), rr, 255, -1)
            roi_masked = cv2.bitwise_and(roi, roi, mask=mask)

            blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            blackhat = cv2.morphologyEx(roi_masked, cv2.MORPH_BLACKHAT, blackhat_kernel)
            thresh = cv2.adaptiveThreshold(blackhat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cleanup_kernel)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            found_line = False

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 3:
                    continue

                rect = cv2.minAreaRect(cnt)
                w, h = rect[1]
                if w == 0 or h == 0:
                    continue

                length = max(w, h)
                thickness = min(w, h)
                aspect_ratio = length / thickness
                if (length >= 8 and length <= 20 and thickness <= 4 and aspect_ratio >= 3):
                    found_line = True
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    box[:, 0] += x1
                    box[:, 1] += y1
                    cv2.drawContours(output, [box], 0, (255, 0, 0), 1)

            if found_line:
                line_count += 1

    cv2.imwrite(output_path, output)

    print(f"Detected circles: {detected_count}")
    print(f"Circles containing thin dark lines: {line_count}")
    print(f"Saved output to: {output_path}")