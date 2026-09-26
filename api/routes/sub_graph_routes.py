from fastapi import APIRouter,Depends,Request,Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from src.core.logger import logger
from api.schemas.chat_schema import TestGenerationRequest
from src.pipelines.graph_runner_pipeline import get_graph_runner_pipeline
from api.middlewares.authentication_middleware import authenticate_only_user




router = APIRouter(dependencies=[Depends(authenticate_only_user)])


@router.get("/test")
async def generate_test_paper(request:Request,payload:TestGenerationRequest=Query()):
    """This is the test generation route which generates test_paper"""

    try:
        logger.info("Entered in the generate_test_paper route")
        logger.info(
            "Test generation request received: total_no_of_questions=%s, level=%s, subject_name=%s, exam_type=%s",
            payload.total_no_of_questions,
            payload.level,
            payload.subject_name,
            payload.exam_type,
        )
        pipeline = get_graph_runner_pipeline()
        logger.info("Graph runner pipeline initialized for test generation")

        response=await pipeline.initiate_test_generation(
                    user_id=request.state.user_id,
                    total_no_of_questions=payload.total_no_of_questions,
                    level=payload.level,
                    subject_name=payload.subject_name,
                    exam_type=payload.exam_type

                )

        logger.info("Test paper generated successfully")
        return JSONResponse(
            content=jsonable_encoder({
                "data": response,
                "success": True,
                "message": "test generated succesfully",
            }),
            status_code=200,
        )


    except Exception as e:
        logger.exception("Test paper generation failed: %s", str(e))
        raise HTTPException(status_code=400,detail={"success":False,"data":None,"message":str(e)})
    