import pytest
from pydantic import ValidationError
from langchain_core.exceptions import OutputParserException

from src.domain.state import (
    OrchestratorOutput,
    QueryGenerationOutput,
    ChatOutput,
    Options,
    Question,
    Questions_generation_schema,
)
from src.domain.config_entities import RetrieverConfig, DataIngestionConfig
from src.utils.langchain_utils import TaggedPydanticOutputParser


class TestSchemasUnit:

    def test_orchestrator_output_bool_coercion(self):
        # Strings should be coerced to boolean
        obj_true = OrchestratorOutput(require_db_search="True")
        assert obj_true.require_db_search is True

        obj_false = OrchestratorOutput(require_db_search="false")
        assert obj_false.require_db_search is False

        # Native booleans should remain unchanged
        obj_native = OrchestratorOutput(require_db_search=True)
        assert obj_native.require_db_search is True

    def test_query_generation_output(self):
        obj = QueryGenerationOutput(queries=["query 1", "query 2"])
        assert len(obj.queries) == 2

    def test_question_schema_valid(self):
        q = Question(
            question_text="What is the speed of light?",
            options=Options(
                option1="3x10^8 m/s",
                option2="3x10^6 m/s",
                option3="3x10^5 m/s",
                option4="3x10^3 m/s",
            ),
            correct="option1",
            explanation="The speed of light in vacuum is approx 300,000 km/s.",
            difficulty="medium",
            points=10,
        )
        assert q.correct == "option1"
        assert q.points == 10

    def test_question_schema_invalid_choice_rejected(self):
        with pytest.raises(ValidationError):
            Question(
                question_text="What is 1+1?",
                options=Options(
                    option1="2",
                    option2="3",
                    option3="4",
                    option4="5",
                ),
                correct="option5",  # Invalid! Must be one of option1..option4
                explanation="1+1=2",
            )

    def test_question_schema_invalid_difficulty_rejected(self):
        with pytest.raises(ValidationError):
            Question(
                question_text="What is 1+1?",
                options=Options(
                    option1="2",
                    option2="3",
                    option3="4",
                    option4="5",
                ),
                correct="option1",
                explanation="1+1=2",
                difficulty="impossible",  # Invalid! Must be easy, medium, or hard
            )

    def test_questions_generation_schema_wrapper(self):
        q = Question(
            question_text="Sample",
            options=Options(option1="A", option2="B", option3="C", option4="D"),
            correct="option1",
            explanation="Sample explanation",
        )
        wrapper = Questions_generation_schema(questions=[q])
        assert len(wrapper.questions) == 1

    def test_retriever_config_defaults(self):
        cfg = RetrieverConfig()
        assert cfg.index_name == "notebooklm-index"
        assert cfg.embeding_dim == 384
        assert cfg.metric == "cosine"
        assert cfg.k == 4

    def test_data_ingestion_config(self):
        cfg = DataIngestionConfig(files_path=["/path/doc.pdf"])
        assert cfg.files_path == ["/path/doc.pdf"]
        assert cfg.chunk_size > 0
        assert cfg.chunk_overlap > 0


class TestTaggedPydanticOutputParserUnit:

    def test_format_instructions_contains_tag_and_schema(self):
        parser = TaggedPydanticOutputParser(pydantic_object=OrchestratorOutput, tag_name="output")
        instructions = parser.get_format_instructions()
        assert "<output>" in instructions
        assert "</output>" in instructions
        assert "require_db_search" in instructions

    def test_parse_valid_tagged_json(self):
        parser = TaggedPydanticOutputParser(pydantic_object=OrchestratorOutput, tag_name="output")
        text = "<output>{\"require_db_search\": true}</output>"
        result = parser.parse(text)
        assert isinstance(result, OrchestratorOutput)
        assert result.require_db_search is True

    def test_parse_with_markdown_code_fences_inside_tags(self):
        parser = TaggedPydanticOutputParser(pydantic_object=OrchestratorOutput, tag_name="output")
        text = "<output>\n```json\n{\"require_db_search\": false}\n```\n</output>"
        result = parser.parse(text)
        assert result.require_db_search is False

    def test_parse_fallback_json_block_extraction(self):
        parser = TaggedPydanticOutputParser(pydantic_object=OrchestratorOutput, tag_name="output")
        text = "Here is the result: {\"require_db_search\": true} Hope that helps!"
        result = parser.parse(text)
        assert result.require_db_search is True

    def test_parse_invalid_content_raises_output_parser_exception(self):
        parser = TaggedPydanticOutputParser(pydantic_object=OrchestratorOutput, tag_name="output")
        text = "No json here at all, just plain text."
        with pytest.raises(OutputParserException):
            parser.parse(text)

    @pytest.mark.asyncio
    async def test_aparse_wrapper(self):
        parser = TaggedPydanticOutputParser(pydantic_object=OrchestratorOutput, tag_name="output")
        text = "<output>{\"require_db_search\": true}</output>"
        result = await parser.aparse(text)
        assert result.require_db_search is True
