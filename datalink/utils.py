from cv2 import imencode, imdecode, IMWRITE_JPEG_QUALITY, IMREAD_COLOR
from cv2.typing import MatLike
from typing import Union
import numpy as np


def jpg_encode(image: MatLike, quality: int) -> np.ndarray:
    _, jpg = imencode(".jpg", image, [int(IMWRITE_JPEG_QUALITY), quality])
    return jpg


def jpg_decode(jpg: Union[bytes, np.ndarray]) -> MatLike:
    return imdecode(np.frombuffer(jpg, np.uint8), IMREAD_COLOR)
