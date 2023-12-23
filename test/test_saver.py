from pathlib import Path

from example.models import Detection, Box, Point, Detections
from example.saver import Saver


def test_saver():
    dets: Detections = [Detection(
        absolute_box=(Box(top_left=Point(x=10, y=10), bottom_right=Point(x=10, y=10))),
        relative_box=Box(top_left=Point(x=0.1, y=0.1), bottom_right=Point(x=0.01, y=0.01)),
        score=0.01,
        label_as_int=1,
        label_as_str='person'
    )]
    name = 'file.jpg'
    open('./images/file.jpg', 'w+')

    sut = Saver(Path('./images')).save(name, dets)

    assert Path('./images/file.txt').exists()
    assert Path('./images/file.txt').is_file()

    Path('./images/file.txt').unlink()
