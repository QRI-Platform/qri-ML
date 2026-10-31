from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# app config to load .env variables witch makes it faster than core os module logic
class AppConfig(BaseSettings):
    pine_cone_api_key: str | None = Field(None, validation_alias="PINECONE_API_KEY")
    groq_api_key: str | None = Field(None, validation_alias="GROQ_API_KEY")
    huggingface_api_key: str | None = Field(None, validation_alias="HF_TOKEN")
    postgres_sql_url: str | None = Field(None, validation_alias="POSTGRES_SQL_URL")
    langsmith_api_key: str | None = Field(None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str | None = Field(None, validation_alias="LANGCHAIN_PROJECT")
    langsmith_endpoint: str | None = Field("https://api.smith.langchain.com", validation_alias="LANGCHAIN_ENDPOINT")
    langsmith_tracing: bool | None = Field(True, validation_alias="LANGCHAIN_TRACING_V2")
    langfuse_secret_key: str | None = Field(None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_public_key: str | None = Field(None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_base_url: str | None = Field(None, validation_alias="LANGFUSE_BASE_URL")
    langfuse_host: str | None = Field(None, validation_alias="LANGFUSE_HOST")
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_app_config() -> AppConfig:
    return AppConfig()
