from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ORCHESTRATOR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Analyze the conversation history and user query to decide if vector database retrieval is required.\n\n"
        "RULES:\n"
        "- Set `require_db_search = true` if the query requests facts, document contents, resumes, backgrounds, or specific details. only set it ture when user has uploaded content is true \n has user uploaded contnet {has_documents}"
        "- Set `require_db_search = false` ONLY for casual greetings (e.g., 'hi', 'thanks') or self-contained general statements."
    ),
    MessagesPlaceholder(variable_name="messages")
])

QUERY_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert search query optimizer for vector retrieval.\n\n"
        "TASKS:\n"
        "1. Resolve implicit pronouns/references (e.g., 'his experience', 'that project') using conversation history.\n"
        "2. Generate 1 to 3 distinct, concise search queries targeting key concepts for vector retrieval.\n"
        "Return queries adhering strictly to the output schema."
    ),
    MessagesPlaceholder(variable_name="messages")
])

SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert conversation summarizer.\n\n"
        "GUIDELINES:\n"
        "- Summarize the conversation history in at most {no_of_words} words.\n"
        "- Retain essential context, key entities, decisions, user preferences, and ongoing tasks without adding new facts."
    ),
    MessagesPlaceholder(variable_name="messages")
])

SUMMARY_NODE_PROMPT = SUMMARIZER_PROMPT

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI assistant equipped with RAG, calculations, and memory capabilities.\n\n"
        "--- LONG-TERM USER MEMORIES ---\n"
        "{user_memories}\n\n"
        "--- RETRIEVED CONTEXT ---\n"
        "{context}\n\n"
        "INSTRUCTIONS:\n"
        "1. Answer clearly in markdown format using the provided context and memories.\n"
        "2. Keep responses concise (under {max_words} words)."
    ),
    MessagesPlaceholder(variable_name="messages"),
])