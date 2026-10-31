from fastapi import APIRouter, Depends, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

from api.middlewares.authentication_middleware import authenticate_only_user
from api.schemas.performance_schema import PerformanceGenerationRequest
from src.pipelines.graph_runner_pipeline import get_graph_runner_pipeline

router = APIRouter(dependencies=[Depends(authenticate_only_user)])


@router.post("/")
async def performance_generation_route(requests: Request, payload: PerformanceGenerationRequest):
    """This is the solved test paper performance generator route"""
    try:
        user_id: str = getattr(requests.state, "user_id", None)
        pipeline = get_graph_runner_pipeline()

        results = await pipeline.initiate_performance_generation(
            user_id=user_id,
            total_correct_answers=payload.total_correct_answers,
            total_questions=payload.total_questions,
            solved_test_paper=payload.solved_test_paper,
        )

        return JSONResponse(
            content=jsonable_encoder({
                "data": results,
                "success": True,
                "message": "the performance generater successfully",
            }),
            status_code=200,
        )
    except Exception as e:
        raise HTTPException(status_code=400,detail={"data":str(e),"success":False,"message":"performance generation failed"})



