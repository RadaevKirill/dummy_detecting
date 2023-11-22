import cv2  # type: ignore

from example.models import GREEN, Image, Detections


class OpenCVPainter:
    def paint(self, image: Image, detections: Detections) -> Image:
        for det in detections:
            x1, y1, x2, y2 = det.absolute_box.as_tuple
            cv2.rectangle(image, (x1, y1), (x2, y2), GREEN, 2)

        return image
