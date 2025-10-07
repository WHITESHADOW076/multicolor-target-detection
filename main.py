import cv2
import numpy as np

def detect_color_targets(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Color ranges in HSV - red has two ranges for hue wrap-around
    color_ranges = {
        'Red': [ (np.array([0, 120, 70]), np.array([10, 255, 255])),
                 (np.array([170, 120, 70]), np.array([180, 255, 255])) ],
        'Green': [ (np.array([36, 50, 70]), np.array([89, 255, 255])) ],
        'Blue': [ (np.array([90, 50, 70]), np.array([128, 255, 255])) ]
    }

    output_frame = frame.copy()
    kernel = np.ones((5,5), np.uint8)

    for color_name, ranges in color_ranges.items():
        # Combine masks for colors with multiple ranges (like red)
        mask = None
        for lower, upper in ranges:
            part_mask = cv2.inRange(hsv, lower, upper)
            mask = part_mask if mask is None else cv2.bitwise_or(mask, part_mask)

        # Noise removal
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                x, y, w, h = cv2.boundingRect(cnt)
                # Draw a small rectangle box: smaller and thicker for clarity
                cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Draw a smaller green box tightly around contour center for emphasis
                cx, cy = x + w//2, y + h//2
                small_box_size = 10
                cv2.rectangle(output_frame, 
                              (cx-small_box_size, cy-small_box_size), 
                              (cx+small_box_size, cy+small_box_size), 
                              (0, 255, 0), 2)
                cv2.putText(output_frame, f"{color_name} Target", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return output_frame

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or set filename for video

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detect_color_targets(frame)

        cv2.imshow('Multi-Color Target Detection', result)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
