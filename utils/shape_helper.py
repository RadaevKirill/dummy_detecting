from example.models import Box, Coordinate, Point
from typing import Optional


def intersect(box1: Box[Coordinate], box2: Box[Coordinate]) -> Optional[Box[Coordinate]]:
    top_left_x = max(box1.top_left.x, box2.top_left.x)
    top_left_y = max(box1.top_left.y, box2.top_left.y)
    bottom_right_x = min(box1.bottom_right.x, box2.bottom_right.x)
    bottom_right_y = min(box1.bottom_right.y, box2.bottom_right.y)

    if top_left_x < bottom_right_x and top_left_y < bottom_right_y:
        return Box(
            top_left=Point(top_left_x, top_left_y),
            bottom_right=Point(bottom_right_x, bottom_right_y)
        )
    else:
        return None  # Нет пересечения
