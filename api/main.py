from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.logger import logger
from src.core.dependencies import warmup_dependencies
from src.pipelines import get_pipeline
from api.routes.graph_routes import router as graph_router
from api.routes.user_routes import router as UserRouter


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server startup: warming up all singletons...")
    warmup_dependencies()
    get_pipeline()
    logger.info("All singletons initialized — server is ready")
    yield
    logger.info("Server shutdown")


app = FastAPI(lifespan=lifespan)
app.include_router(graph_router, prefix="/api/v1/graph")
app.include_router(UserRouter,prefix="/api/v1/user")
