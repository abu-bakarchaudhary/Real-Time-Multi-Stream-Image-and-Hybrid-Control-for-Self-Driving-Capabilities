import cv2
import numpy as np


class ObjectDetector:

    def __init__(self, model_cfg, model_weights, class_names):

        # Load YOLO network
        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)

        # Get output layer names
        self.layer_names = self.net.getLayerNames()

        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        # COCO class labels (car, person, bike, etc.)
        self.classes = class_names

        # Detection thresholds
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4


    def detect_objects(self, frame):

        input_frame = cv2.resize(frame, (320, 320))

        r = frame.shape[0]
        c = frame.shape[1]

        # =====================================================
        # STEP 1: Create blob from image (YOLO input format)
        # =====================================================
        blob = cv2.dnn.blobFromImage(input_frame, 1/255.0, (320, 320),
                                     swapRB=True, crop=False)

        self.net.setInput(blob)

        # =====================================================
        # STEP 2: Forward pass through network
        # =====================================================
        outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        # =====================================================
        # STEP 3: Process detections
        # =====================================================
        for output in outputs:

            for detection in output:

                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.conf_threshold:

                    # scale bounding box back to image size
                    # YOLO gives normalized coordinates (0–1 relative to 320x320 input)
                    center_x = int(detection[0] * 320)
                    center_y = int(detection[1] * 320)
                    w = int(detection[2] * 320)
                    h = int(detection[3] * 320)

                    # scale from 320x320 → original frame size
                    scale_x = c / 320
                    scale_y = r / 320

                    center_x = int(center_x * scale_x)
                    center_y = int(center_y * scale_y)
                    w = int(w * scale_x)
                    h = int(h * scale_y)

                    # convert center → top-left
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # =====================================================
        # STEP 4: Non-Max Suppression (remove duplicates)
        # =====================================================
        indices = cv2.dnn.NMSBoxes(boxes, confidences,
                                   self.conf_threshold,
                                   self.nms_threshold)

        detected_objects = []

        for i in indices:

            i = i[0] if isinstance(i, (list, np.ndarray)) else i

            x, y, w, h = boxes[i]

            label = str(self.classes[class_ids[i]])
            conf = confidences[i]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 255), 2)

            conf_percent = int(conf * 100)

            text = f"{label} {conf_percent}%"

            cv2.putText(frame, text,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

            detected_objects.append({
                "label": label,
                "confidence": conf,
                "box": (x, y, w, h)
            })

        return frame, detected_objects