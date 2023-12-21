import time
from pathlib import Path

from example.person_detector import PersonDetector
from example.decoder import OpenCVDecoder
from example.saver import Saver


class Example:
    def __init__(self, detector, decoder, saver, folder: Path) -> None:
        self._detector = detector
        self._decoder = decoder
        self._saver = saver
        self._folder = folder

    def run(self) -> None:
        while True:
            if len([self._folder.rglob('*.jpg')]) > 0:
                for frame in self._decoder.decode():
                    detections = self._detector.detect(image=frame.image)
                    self._saver.save(detections=detections, name=frame.name)
            else:
                time.sleep(2)


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
            decoder=OpenCVDecoder(Path('./images')),
            saver=Saver(Path('./images')),
            folder=Path('./images')
            ).run()
