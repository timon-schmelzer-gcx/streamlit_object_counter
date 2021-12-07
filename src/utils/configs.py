import enum
import typing
from dataclasses import dataclass

class IMAGES(enum.Enum):
    DOTS = 'images/dots.png'
    BIRDS = 'images/birds.jpeg'
    BIRDS2 = 'images/birds_2.png'
    KEYBOARD = 'images/keyboard.png'

class RunningModes(enum.Enum):
    EXAMPLES = 'View Examples'
    UPLOAD = 'Upload Own Image'

@dataclass
class DetectionConfig():
    thresh_val: typing.Optional[int]=None
    thresh_type: typing.Optional[int]=None
    thresh_maxval: typing.Optional[int]=None

    contour_mode: typing.Optional[int]=None
    contour_method: typing.Optional[int]=None

    min_contour_length: typing.Optional[int]=None
    max_contour_length: typing.Optional[int]=None

default_config = DetectionConfig(
    thresh_val=150,
    thresh_type=1,
    thresh_maxval=255,
    contour_mode=1,
    contour_method=2,
    min_contour_length=0,
    max_contour_length=100
)

birds_config = DetectionConfig(
    thresh_val=62,
    thresh_type=3,
    thresh_maxval=255,
    contour_mode=1,
    contour_method=1,
    min_contour_length=2,
    max_contour_length=40
)

birds2_config = DetectionConfig(
    thresh_val=125,
    thresh_type=1,
    thresh_maxval=255,
    contour_mode=1,
    contour_method=2,
    min_contour_length=0,
    max_contour_length=100
)

dots_config = DetectionConfig(
    thresh_val=163,
    thresh_type=3,
    thresh_maxval=255,
    contour_mode=1,
    contour_method=3,
    min_contour_length=6,
    max_contour_length=20
)

keyboard_config = DetectionConfig(
    thresh_val=33,
    thresh_type=1,
    thresh_maxval=255,
    contour_mode=0,
    contour_method=3,
    min_contour_length=9,
    max_contour_length=41
)
