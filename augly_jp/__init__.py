from .base_augmenter import Augmenter
from typing import List, Union, Iterable
import numpy as np

Element = Union[str, np.ndarray]
Data = Union[Element, List[Element]]
