from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langfuse import Langfuse

lf = Langfuse()

def load_prompt(name):
    return lf.get_prompt(name).get_langchain_prompt()

ORCHESTRATOR_PROMPT = ChatPromptTemplate.from_messages(load_prompt("ORCHESTRATOR_PROMPT"))

QUERY_GENERATION_PROMPT = ChatPromptTemplate.from_messages(load_prompt("QUERY_GENERATION_PROMPT"))

SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages(load_prompt("SUMMARIZER_PROMPT"))

SUMMARY_NODE_PROMPT = SUMMARIZER_PROMPT

CHAT_PROMPT = ChatPromptTemplate.from_messages(load_prompt("CHAT_PROMPT"))