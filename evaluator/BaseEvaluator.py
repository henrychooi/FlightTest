from abc import ABC, abstractmethod
from typing import Optional
import os

class BaseEvaluator(ABC):
    def __init__(self, model_path: os.PathLike, data_path: os.PathLike, prompt_path: Optional[os.PathLike]):
        super().__init__()
        self.model_path = model_path
        self.model_name = os.path.basename(self.model_path)
        self.data_path = data_path
        self.prompt_path = None or prompt_path

    @abstractmethod
    async def evaluate_model(self):
        pass

