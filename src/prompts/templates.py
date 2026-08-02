from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ORCHESTRATOR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an orchestrator that decides whether a user query requires searching a document database.\n"
        "Default to require_db_search = true for any questions asking about documents, resumes, PDFs, names, details, background, facts, summaries, or specific information.\n"
        "Only set require_db_search to false if the user message is purely a casual greeting (e.g., 'hi', 'hello', 'how are you', 'thanks') with no request for information."
    ),
    MessagesPlaceholder(variable_name="messages")
])
QUERY_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a query generator. Given the user's question, generate 1-3 precise search queries "
               "that will help retrieve the most relevant documents from a vector database to answer the question.\n"
               "Return only the queries matching the required output schema."),
    MessagesPlaceholder(variable_name="messages")
])

SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    (
        "human",
        "If existing summary is present, extend it by incorporating the new messages above. "
        "Otherwise, create a new summary.\n"
        "You are an expert conversation summarizer. "
        "Summarize the conversation in at most {no_of_words} words "
        "while preserving all important context, decisions, user preferences, "
        "tasks, and ongoing discussions. Do not add any new information."
    )
])
CHAT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI assistant equipped with a Retrieval-Augmented Generation (RAG) system.\n"
        "You must answer the user's question clearly and accurately using the provided retrieved database context.\n\n"
        "--- LONG-TERM USER MEMORIES ---\n"
        "{user_memories}\n\n"
        "--- RETRIEVED VECTOR DB CONTEXT ---\n"
        "{context}\n\n"
        "INSTRUCTIONS:\n"
        "1. Prioritize the retrieved vector DB context to answer factual questions.\n"
        "2. If personal preferences or user details are mentioned, extract 'memory_key' and 'memory_value', otherwise set them to null."
    ),
    MessagesPlaceholder(variable_name="messages"),
])

SUMMARY_NODE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert conversation summarizer. "
        "Summarize the given conversation in at most {no_of_words} words "
        "while preserving all important context, decisions, user preferences, "
        "tasks, and ongoing discussions. Do not add any new information."
    ),
    MessagesPlaceholder(variable_name="messages"),
])
