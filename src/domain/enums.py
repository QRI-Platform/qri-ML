from abc import ABC, abstractmethod


# abc method for pipeline definition (in python, used for abstraction)
class Pipeline(ABC):
    def __init__(self):
        pass

    @abstractmethod
    async def initiate(self):
        pass
