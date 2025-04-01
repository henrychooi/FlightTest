from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    @abstractmethod
    async def evaluate_model(self)