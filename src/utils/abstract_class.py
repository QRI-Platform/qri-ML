from abc import abstractmethod,ABC


class Pipeline(ABC):
    def __init__(self):
        pass

    @abstractmethod
    async def initiate(self):
        pass