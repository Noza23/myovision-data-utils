from segment_anything.modeling import Sam
from typing import Union, TYPE_CHECKING
if TYPE_CHECKING:
    from .image import Image

MODEL: Union[Sam, None] = None
IMAGE: Union['Image', None] = None