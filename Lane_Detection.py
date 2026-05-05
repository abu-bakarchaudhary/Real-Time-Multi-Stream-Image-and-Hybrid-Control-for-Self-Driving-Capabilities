import cv2
import numpy as np


def region_of_interest(img):
    r = img.shape[0]
    c = img.shape[1]

    mask = np.zeros((r, c), dtype=np.uint8)

    # Adjusted polygon to capture the wider curb lines seen in the screenshot
    polygon = np.array([[
        (int(0.02 * c), r),
        (int(0.98 * c), r),
        (int(0.65 * c), int(0.40 * r)),
        (int(0.35 * c), int(0.40 * r))
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)

    roi_img = cv2.bitwise_and(img, mask)

    return roi_img


def lane_detection(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Lowered Canny thresholds slightly to better pick up the yellow curb edges
    edges = cv2.Canny(blur, 40, 120)

    roi = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi / 180,
        threshold=40, # Reduced threshold to catch segments of the curb
        minLineLength=40,
        maxLineGap=150
    )

    line_img = np.zeros_like(frame)

    left_x = []
    right_x = []

    if lines is not None:

        left_lines = []
        right_lines = []

        for i in range(len(lines)):

            x1, y1, x2, y2 = lines[i][0]

            if (x2 - x1) != 0:
                slope = (y2 - y1) / (x2 - x1)
            else:
                slope = 999

            # Widened slope range: Curb lines are often less steep (between 0.3 and 0.8)
            if abs(slope) < 0.2 or abs(slope) > 2.0:
                continue

            # Removed the strict horizontal position check to allow lines near the edges
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
                left_x.append(x1)
                left_x.append(x2)

            elif slope > 0:
                right_lines.append((x1, y1, x2, y2))
                right_x.append(x1)
                right_x.append(x2)

        if len(left_lines) > 0:
            left_avg_line = np.mean(left_lines, axis=0).astype(int)
            x1, y1, x2, y2 = left_avg_line
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 4)

        if len(right_lines) > 0:
            right_avg_line = np.mean(right_lines, axis=0).astype(int)
            x1, y1, x2, y2 = right_avg_line
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 4)

    frame_center = frame.shape[1] // 2

    lane_center = frame_center

    if len(left_x) > 0 and len(right_x) > 0:

        left_avg = int(np.mean(left_x))
        right_avg = int(np.mean(right_x))

        lane_center = (left_avg + right_avg) // 2

    offset = lane_center - frame_center

    if offset > 20:
        direction = "RIGHT"
    elif offset < -20:
        direction = "LEFT"
    else:
        direction = "FORWARD"

    output = cv2.addWeighted(frame, 0.8, line_img, 1, 1)

    cv2.rectangle(output,
                  (0, int(0.82 * frame.shape[0])),
                  (frame.shape[1], frame.shape[0]),
                  (0, 0, 0),
                  -1)

    cv2.putText(output,
                f"TURN {direction}" if direction != "FORWARD" else "FORWARD",
                (180, int(0.93 * frame.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                (0, 255, 255),
                4)

    lane_status = "Detected" if len(left_x) > 0 or len(right_x) > 0 else "No Lanes Detected"

    return output, direction, lane_status