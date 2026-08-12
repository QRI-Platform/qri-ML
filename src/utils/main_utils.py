import json
from src.core.exceptions import MyException

# loading json to python dict

async def load_json_file(file_path: str) -> dict:
    """Load a JSON file and return its contents as a dictionary."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        raise MyException(f"File not found: {file_path}")