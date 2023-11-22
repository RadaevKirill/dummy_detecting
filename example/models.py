import time
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Dict, Generic, List, Tuple, TypeVar, Union

import numpy as np
from nptyping import Float32, Int32, NDArray, Shape, UInt8

Color = namedtuple('Color', 'B G R')
BLUE = Color(255, 0, 0)
GREEN = Color(0, 255, 0)
RED = Color(0, 0, 255)
MAGENTA = Color(255, 0, 255)

ImageRGB = NDArray[Shape['* height, * width, 3 rgb'], UInt8]
ImageBGR = NDArray[Shape['* height, * width, 3 bgr'], UInt8]
Image = Union[ImageRGB, ImageBGR]


Coordinate = TypeVar('Coordinate', int, float)

@dataclass
class Frame:
    number_image: int
    image: Image


@dataclass
class Point(Generic[Coordinate]):
    x: Coordinate  # noqa: VNE001
    y: Coordinate  # noqa: VNE001

    @property
    def as_tuple(self) -> Tuple[Coordinate, Coordinate]:
        return self.x, self.y

    def __add__(self, right: 'Point[Coordinate]') -> 'Point[Coordinate]':
        if not isinstance(right, type(self)):
            raise ValueError("The 'right' must be an instance of the Point")

        return Point(self.x + right.x, self.y + right.y)

    def __sub__(self, right: 'Point[Coordinate]') -> 'Point[Coordinate]':
        if not isinstance(right, Point):
            raise ValueError("The 'right' must be an instance of the Point")

        return Point(self.x - right.x, self.y - right.y)


Points = List[Point[Coordinate]]


@dataclass
class Box(Generic[Coordinate]):
    top_left: Point[Coordinate]
    bottom_right: Point[Coordinate]

    @property
    def as_tuple(self) -> Tuple[Coordinate, Coordinate, Coordinate, Coordinate]:
        return self.top_left.x, self.top_left.y, self.bottom_right.x, self.bottom_right.y

    @property
    def center(self) -> Point[Coordinate]:
        return Point(
            type(self.top_left.x)((self.top_left.x + self.bottom_right.x) / 2),
            type(self.top_left.x)((self.top_left.y + self.bottom_right.y) / 2),
        )


Boxes = List[Box]  # type: ignore
AbsoluteNDArrayBoxes = NDArray[Shape['*, [left, top, right, bottom]'], Int32]
RelativeNDArrayBoxes = NDArray[Shape['*, [left, top, right, bottom]'], Float32]
NDArrayBoxes = Union[AbsoluteNDArrayBoxes, RelativeNDArrayBoxes]
AbsoluteNDArrayCoordinates = NDArray[Shape['*, [x, y]'], Int32]
RelativeNDArrayCoordinates = NDArray[Shape['*, [x, y]'], Float32]


@dataclass
class Detection:
    absolute_box: Box[int]
    relative_box: Box[float]
    score: float
    label_as_str: str
    label_as_int: int

Detections = List[Detection]

