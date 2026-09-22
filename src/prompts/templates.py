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

AGENT_PROMPT = ChatPromptTemplate.from_messages(load_prompt("AGENT_PROMPT"))







# ============================= Test Paper Generation SubGraph prompts ============================
TEST_PAPER_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a professional exam question paper generator.

Generate a test paper based strictly on the provided subject, exam type, difficulty distribution, and total number of questions.

Requirements:
- Generate exactly {total_no_of_questions} questions.
- Easy questions: {no_of_easy_questions}
- Medium questions: {no_of_medium_questions}
- Hard questions: {no_of_hard_questions}
- Question difficulty level: {level}
- Subject: {subject_name}
- Exam type: {exam_type}
- Ensure the questions are relevant to the subject and appropriate for the specified exam type.
- Maintain the requested difficulty distribution exactly.
- Do not generate duplicate or substantially similar questions.
- Do not include questions outside the specified subject.
- Ensure each question is clear, unambiguous, and self-contained.
- Return only the generated test paper in the required output format."""
        )
    ]
)



