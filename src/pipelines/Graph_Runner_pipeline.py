from src.utils.abstract_class import Pipeline
from src.graphs.builder import graph
from src.models.workflow_models import State


class GraphRunnerPipeline(Pipeline):
    def __init__(self):
        self.graph = graph
        pass

    async def initiate(self,thread_id,file_paths):
        state = State(
            user_id=thread_id,
            file_paths=file_paths
        )

        config={"configurable":{"thread_id":thread_id}}
        async for s in self.graph.astream(state,config=config):
            yield s

