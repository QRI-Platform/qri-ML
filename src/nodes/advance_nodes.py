
from src.exception import MyException
import sys
from src.models.workflow_models import State
from src.logger import logger
from src.llm.llm_loader import llm
from src.constants import NO_OF_LAST_MESSAGES_TO_KEEP
from langchain_core.messages import RemoveMessage,HumanMessage


async def summerizer(state: State):
    try:
        logger.info("Entered in the summerization state")

        summary = state.get("summary", "")
        messages = state.get("messages", [])

        messages_to_summarize = messages[:-NO_OF_LAST_MESSAGES_TO_KEEP]

        if not messages_to_summarize:
            logger.info("Not enough messages to summarize.")
            return {}

        if summary:
            summary_prompt = (
                f"Existing conversation summary:\n{summary}\n\n"
                "Extend the summary by incorporating the new messages above. "
                "Keep it concise and focus on key context."
            )
        else:
            summary_prompt = (
                "Create a concise summary of the conversation above, "
                "focusing on key context, decisions, and user preferences."
            )

        summary_messages = messages_to_summarize + [HumanMessage(content=summary_prompt)]
        response = await llm.ainvoke(summary_messages)

        delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]

        logger.info("Successfully updated conversation summary.")

        return {
            "summary": response.content,
            "messages": delete_messages
        }

    except Exception as e:
        logger.error(f"Error during summarization: {str(e)}")
        raise MyException(e, sys)

    