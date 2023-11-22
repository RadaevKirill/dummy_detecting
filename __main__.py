import cv2
import numpy as np
import streamlit as st

from example.person_detector import PersonDetector
from example.painter import OpenCVPainter


class Example:
    def __init__(
            self,
            detector,
            painter
    ) -> None:
        self._detector = detector
        self._painter = painter

    def run(self) -> None:
        # for frame in self._decoder.decode():
        #     detections = self._detector.detect(image=frame.image)
        #     self._saver.save(frame=frame, detections=detections)
        #     self._visualizer.visualize(image=frame.image, detections=detections)

        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            image = cv2.imdecode(np.frombuffer(img_file_buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            detections = self._detector.detect(image=image)
            image_with_det = self._painter.paint(image=image, detections=detections)
            st.image(image_with_det)


if __name__ == '__main__':
    Example(detector=PersonDetector(device='CPU',
                                    performance_hint='LATENCY',
                                    execution_mode_hint='PERFORMANCE',
                                    inference_precision_hint='f32',
                                    model='./assets/person-detection/person-detection-0202-fp32.xml',
                                    weights='./assets/person-detection/person-detection-0202-fp32.xml',
                                    input_type='float32',
                                    input_dims_order='bchw',
                                    swap_rb=False,
                                    height=512,
                                    width=512,
                                    score_threshold=0.6,
                                    nms_threshold=0.4),
            painter=OpenCVPainter()
            ).run()
