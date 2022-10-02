import pandas as pd
import numpy as np
from typing import List, Tuple

from .abstract_model_parser import AbstractModelParser


class BaseModelParser(AbstractModelParser):

    def __init__(self, model, model_type, task: str):
        super().__init__(model, model_type)
        self.task = task
