import cv2  # type: ignore
import numpy as np
from nptyping import Float32, NDArray, Shape

from example.models import Box, Detection, Detections, Image, Point
from example.openvino_adapter import OpenvinoAdapterMixin


Output = NDArray[Shape['1, 1, 200 max_total_detections, [image_id, label, score, x_min, y_min, x_max, y_max]'], Float32]
OutputNew = NDArray[Shape['1, N num detection, [x_min, y_min, x_max, y_max]'], Float32]
FilteredOutput = NDArray[Shape['* num_detections, [image_id, label, score, x_min, y_min, x_max, y_max]'], Float32]
Boxes = NDArray[Shape['* num_detections, [x_min, y_min, x_max, y_max]'], Float32]


class PersonDetector(OpenvinoAdapterMixin):
    def _post_processing(self, output: Output, image_height: int, image_width: int) -> Detections:
        filtered_out: FilteredOutput = output[0, 0, output[0, 0, :, 2] > self._score_threshold]
        boxes: Boxes = (filtered_out[:, 3:] * ([image_width, image_height] * 2)).astype(int)
        boxes = np.where(boxes < 0, 0, boxes)
        detections: Detections = []

        for (_, label, score, rx1, ry1, rx2, ry2), (ax1, ay1, ax2, ay2) in zip(filtered_out, boxes):
            detections.append(
                Detection(
                    absolute_box=Box[int](top_left=Point(x=ax1, y=ay1), bottom_right=Point(x=ax2, y=ay2)),
                    relative_box=Box[float](top_left=Point(x=rx1, y=ry1), bottom_right=Point(x=rx2, y=ry2)),
                    score=score,
                    label_as_str='person',
                    label_as_int=label,
                ),
            )

        return detections

    def detect(self, image: Image) -> Detections:
        detections: Detections = self._predict(image=image)
        return detections
