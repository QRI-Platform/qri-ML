import os
import shutil
import sys
from fastapi import Request, UploadFile
from src.constants import PUBLIC_TEMP_DIR
from src.exception import MyException
from src.logger import logger

os.makedirs(PUBLIC_TEMP_DIR, exist_ok=True)


async def multer_middleware(request: Request, file: UploadFile = None):
    file_path: str = ""
    try:
        if not file or not file.filename:
            logger.info("multer_middleware: no file in request")
            yield ""
            return

        thread_id = request.state.thread_id
        dest_dir = os.path.join(PUBLIC_TEMP_DIR, str(thread_id))
        os.makedirs(dest_dir, exist_ok=True)

        file_path = os.path.join(dest_dir, file.filename)
        logger.info("multer_middleware: saving file to %s", file_path)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.info("multer_middleware: file saved successfully at %s", file_path)
        
        # Pass file path to route/endpoint
        yield file_path

    except Exception as e:
        logger.error("multer_middleware: error saving file: %s", str(e))
        raise MyException(e, sys)

    finally:
        # Clean up file/folder after request processing is completed
        if file_path and os.path.exists(file_path):
            try:
                shutil.rmtree(os.path.dirname(file_path))
                logger.info("multer_middleware: cleaned up temp file %s", file_path)
            except Exception as clean_err:
                logger.error("multer_middleware: failed to clean up %s: %s", file_path, str(clean_err))