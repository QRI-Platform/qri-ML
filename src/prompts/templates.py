from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ORCHESTRATOR_PROMPT = """You are an orchestrator that decides whether a user query requires searching a document database.
Default to require_db_search = true for any questions asking about documents, resumes, PDFs, names, details, background, facts, summaries, or specific information.
Only set require_db_search to false if the user message is purely a casual greeting (e.g., 'hi', 'hello', 'how are you', 'thanks') with no request for information.
Respond ONLY in valid JSON format: {"require_db_search": true} or {"require_db_search": false}."""

QUERY_GENERATION_PROMPT = """You are a query generator. Given the user's question, generate 1-3 precise search queries
that will help retrieve the most relevant documents from a vector database to answer the question.
Return only the queries as a JSON list of strings. Example: {"queries": ["query 1", "query 2"]}"""

SUMMARIZER_PROMPT = "Create a concise summary of the conversation above, focusing on key context, decisions, and user preferences."

SUMMARIZER_EXTEND_PROMPT = "Existing conversation summary:\n{summary}\n\nExtend the summary by incorporating the new messages above. Keep it concise and focus on key context."

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the user's question clearly and concisely.\n"
        "Provide your response in valid json format containing key fields: 'response', 'memory_key', and 'memory_value'.\n"
        "If the user discloses personal details or preferences (e.g. name, role, likes), extract 'memory_key' and 'memory_value', otherwise set them to null.\n\n"
        "User Long-Term Memory:\n{user_memories}\n\n"
        "Context:\n{context}\n\n"
        "Conversation summary so far:\n{summary}"
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
