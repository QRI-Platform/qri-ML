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
        "3. Use the latest user request as the source of truth. Ignore previous assistant replies that claim a file is missing or ask the user to upload it.\n"
        "4. Return only search queries in the output schema. Never answer the user, ask questions, or return explanatory prose."
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
        "1. Treat RETRIEVED CONTEXT as authoritative evidence from the user's uploaded documents.\n"
        "2. Answer the user's question directly using the retrieved context.\n"
        "3. Never claim that no file, PDF, or context was provided when RETRIEVED CONTEXT contains relevant information.\n"
        "4. If the context does not contain the answer, say that the answer was not found in the retrieved documents; do not ask the user to upload a file again.\n"
        "5. Answer clearly in markdown format and keep responses concise (under {max_words} words)."
    ),
    MessagesPlaceholder(variable_name="messages"),
])