from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional
from functools import lru_cache

class AppConfig(BaseSettings):
    pine_cone_api_key: Optional[str] = Field(None, validation_alias="PINECONE_API_KEY")
    groq_api_key: Optional[str] = Field(None, validation_alias="GROQ_API_KEY")
    mlflow_api_key: Optional[str] = Field(None, validation_alias="MLFLOW_API_KEY")
    huggingface_api_key: Optional[str] = Field(None, validation_alias="HUGGINGFACE_API_KEY")
    dagshub_owner: str = Field("vanshsharma7832", validation_alias="DAGSHUB_OWNER")
    dagshub_repo: str = Field("E-Commerce-Chatbot-recommendation", validation_alias="DAGSHUB_REPO")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

@lru_cache
def getAppConfig() -> AppConfig:
    return AppConfig()

app_config: AppConfig = getAppConfig()
