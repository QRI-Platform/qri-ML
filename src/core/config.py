from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional
from functools import lru_cache


class AppConfig(BaseSettings):
    pine_cone_api_key: Optional[str] = Field(None, validation_alias="PINECONE_API_KEY")
    groq_api_key: Optional[str] = Field(None, validation_alias="GROQ_API_KEY")
    huggingface_api_key: Optional[str] = Field(None, validation_alias="HF_TOKEN")
    postgres_sql_url: Optional[str] = Field(None, validation_alias="POSTGRES_SQL_URL")
    langsmith_api_key: Optional[str] = Field(None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: Optional[str] = Field(None, validation_alias="LANGCHAIN_PROJECT")
    langsmith_endpoint: Optional[str] = Field("https://api.smith.langchain.com", validation_alias="LANGCHAIN_ENDPOINT")
    langsmith_tracing: Optional[bool] = Field(True, validation_alias="LANGCHAIN_TRACING_V2")
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_app_config() -> AppConfig:
    return AppConfig()
