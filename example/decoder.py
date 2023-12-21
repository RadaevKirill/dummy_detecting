import time
from pathlib import Path

import cv2

from example.models import Image, Frame


class OpenCVDecoder:
    def __init__(self, folder: Path):
        self._folder = folder

    def decode(self) -> Frame:
        time.sleep(1)
        images = self._folder.rglob('*.jpg')
        for path in images:
            image = cv2.imread(str(path))
            yield Frame(number_image=0, image=image, name=path.name)
