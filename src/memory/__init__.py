from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.postgres import PostgresSaver
from src.config.app_config import app_config
serde = EncryptedSerializer.from_pycryptodome_aes()
checkpointer = PostgresSaver.from_conn_string(app_config.postgres_sql_url, serde=serde)
checkpointer.setup()