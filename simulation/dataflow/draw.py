from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class DrawText:
    text: str
    color: Tuple[int, int, int]
    view: str = "info"



