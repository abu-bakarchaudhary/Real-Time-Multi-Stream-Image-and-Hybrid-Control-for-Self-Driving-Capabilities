import cv2
import time
import os
from dataclasses import dataclass

from object_detection import ObjectDetector
from Lane_Detection import lane_detection


@dataclass
class FramePackage:
    frame_id: int
    front_frame: any
    side_frame: any
    timestamp: float


class VideoPipelineManager:

    def __init__(self, front_video_path, side_video_path=None, width=640, height=360):

        self.front_cap = cv2.VideoCapture(front_video_path)

        if not self.front_cap.isOpened():
            print("Error: Front video not opened. Check video path.")
            exit()

        self.side_cap = cv2.VideoCapture(side_video_path) if side_video_path else None

        self.width = width
        self.height = height
        self.frame_id = 0
        self.prev_time = time.time()

    def _resize(self, frame):
        return cv2.resize(frame, (self.width, self.height))

    def get_frame_package(self):

        ret1, front_frame = self.front_cap.read()

        if not ret1:
            return None

        side_frame = None

        if self.side_cap:
            ret2, side_frame = self.side_cap.read()

            if not ret2:
                side_frame = front_frame.copy()

        front_frame = self._resize(front_frame)

        if side_frame is not None:
            side_frame = self._resize(side_frame)

        current_time = time.time()
        fps = 1 / (current_time - self.prev_time + 1e-6)
        self.prev_time = current_time

        package = FramePackage(
            frame_id=self.frame_id,
            front_frame=front_frame,
            side_frame=side_frame,
            timestamp=current_time
        )

        self.frame_id += 1

        return package, fps

    def release(self):
        self.front_cap.release()

        if self.side_cap:
            self.side_cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    dataset_path = r"E:\Study content\6th SEMESTER\DIGITAL IMAGE PROCESSING (DIP)\PROJECT\SELF DRIVING PROJECT\DATASET\DIP Project Videos"

    video_name = "PXL_20250325_045117252.TS.mp4"

    video_path = os.path.join(dataset_path, video_name)

    pipeline = VideoPipelineManager(video_path)

    lab1_path = r"E:\Study content\6th SEMESTER\DIGITAL IMAGE PROCESSING (DIP)\DIP WORK\LAB1"

    coco_classes = open(os.path.join(lab1_path, "coco.names")).read().strip().split("\n")

    detector = ObjectDetector(
        os.path.join(lab1_path, "yolov3.cfg"),
        os.path.join(lab1_path, "yolov3.weights"),
        coco_classes
    )

    count = 0

    while True:

        result = pipeline.get_frame_package()

        if result is None:
            break

        package, fps = result

        frame = package.front_frame.copy()

        frame, direction, lane_status = lane_detection(frame)

        count += 1

        if count % 2 != 0:
            continue

        objects = []

        if count % 5 == 0:
            frame, objects = detector.detect_objects(frame)

        obstacle_count = len(objects)

        if obstacle_count > 0:
            final_decision = "GO: Proceed "
        else:
            final_decision = "GO: Clear"

        cv2.putText(frame, f"Lane: {lane_status}", (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Steering: {direction}", (40, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"YOLO: {obstacle_count} obstacle(s)", (40, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, final_decision, (40, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

        cv2.putText(frame, f"FPS: {fps:.2f}", (40, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Self Driving System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    pipeline.release()