from dotenv import load_dotenv
load_dotenv()
import sys
import subprocess
from api.main import app

def dev():
    cmd = [
        "opentelemetry-instrument",
        "uvicorn",
        "api.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "7860",
        "--reload",
    ]
    try:
        sys.exit(subprocess.run(cmd).returncode)
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        sys.exit(0)


def start():
    cmd = [
        "opentelemetry-instrument",
        "uvicorn",
        "api.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "7860",
    ]
    try:
        sys.exit(subprocess.run(cmd).returncode)
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        sys.exit(0)

if __name__ == "__main__":
    dev()
    