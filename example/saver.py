from pathlib import Path

from example.models import Detections


class Saver:
    def __init__(self, folder: Path):
        self._folder = folder

    def save(self, name: Path, detections: Detections):
        with open(self._folder / Path(name).with_suffix('.txt'), 'w+') as f:
            for det in detections:
                tlx, tly, brx, bry = det.relative_box.as_tuple
                f.write(str(det.relative_box.as_tuple) + '\n')

        (self._folder / Path(name).with_suffix('.jpg')).unlink()

