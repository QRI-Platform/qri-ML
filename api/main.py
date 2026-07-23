from fastapi import FastAPI
from api.routes.graph_routes import router as graph_router
from api.routes.user_routes import router as UserRouter
app = FastAPI()

app.include_router(graph_router, prefix="/api/v1")
app.include_router(UserRouter,prefix="/api/v1")
