import re

import json
from typing import Generic,TypeVar,Optional,Any

from pydantic import BaseModel, ValidationError
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException
T = TypeVar("T", bound=BaseModel)


class TaggedPydanticOutputParser(BaseOutputParser[T],Generic[T]):
    pydantic_object:type[T]
    tag_name:str = "output"

    def get_format_instructions(self)->str:
        schmea=self.pydantic_object.model_json_schema()

        reduced_schema = {
            k:v for k,v in schmea.items() if k in ["properties", "required", "type"]

        }

        return (
            f"Enclose your structured response inside <{self.tag_name}> and </{self.tag_name}> tags.\n"
            f"Provide ONLY raw JSON matching this schema inside the tags:\n"
            f"{json.dumps(reduced_schema)}"
        )

    

    def parse(self,text:str)->T:
        raw_text = text.strip()

        # 1. Extract content from XML tag
        pattern = rf"<{self.tag_name}>(.*?)</{self.tag_name}>"
        match = re.search(pattern, raw_text, flags=re.DOTALL)
        candidate = match.group(1).strip() if match else raw_text

        # 2. Strip any accidental markdown fences inside tags
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.MULTILINE).strip()

        # 3. JSON Decode & Pydantic Validation
        try:
            json_dict = json.loads(candidate)
            return self.pydantic_object.model_validate(json_dict)
        except (json.JSONDecodeError, ValidationError) as e:
            # Fallback: Agar text me kahin bhi first valid {...} block present ho
            json_fallback = re.search(r"(\{.*\})", candidate, flags=re.DOTALL)
            if json_fallback:
                try:
                    return self.pydantic_object.model_validate(json.loads(json_fallback.group(1)))
                except Exception:
                    pass

            raise OutputParserException(
                f"Failed to parse {self.pydantic_object.__name__} from output: {text}. Reason: {str(e)}"
            )
    
    async def aparse(self,text:str)->str:
        return self.parse(text)


