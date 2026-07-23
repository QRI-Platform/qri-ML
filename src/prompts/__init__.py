from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

ORCHESTRATOR_PROMPT = """You are an orchestrator that decides whether a user query requires searching a document database.
If the question is about documents, files, or specific data that might be in the knowledge base, set require_db_search to true (boolean, not string).
If it is a general conversational question, set require_db_search to false (boolean, not string).
You MUST return require_db_search as a JSON boolean: true or false, never as a string like "true" or "false"."""

QUERY_GENERATION_PROMPT = """You are a query generator. Given the user's question, generate 1-3 precise search queries 
that will help retrieve the most relevant documents from a vector database to answer the question.
Return only the queries as a list."""

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a helpful AI assistant. Answer the user's question accurately and concisely."
        "{context}"
    ),
    HumanMessagePromptTemplate.from_template("{question}")
])
