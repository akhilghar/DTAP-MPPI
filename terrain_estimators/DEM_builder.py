import numpy as np
import dataclasses
from typing import Tuple
import math

class DigElevMap:
    def __init__(self, map_size: Tuple[int, int], resolution: float):
        self.map_size = map_size
        self.resolution = resolution
        self.elevation_map = np.zeros(map_size)