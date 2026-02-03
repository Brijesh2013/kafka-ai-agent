import os
from openai import OpenAI
from rag.retriever import get_retriever
from prompts.kafka_prompt import SYSTEM_PROMPT

# Initialize OpenAI client from environment for safety and clarity
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Set it in your environment or in a .env file (do not commit secrets to source control).")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))


def retrieve_docs(state):
    """Retrieve documents for a query; return a safe fallback on failure.

    Returns a dict with key `retrieved_docs` containing the concatenated docs or a
    clear fallback message so the app can continue running.
    """
    try:
        retriever = get_retriever()
        docs = retriever.get_relevant_documents(state["user_query"])
        content = "\n".join(doc.page_content for doc in docs)
        return {"retrieved_docs": content}
    except Exception as e:
        # Log the exception server-side but return a safe message to the user flow
        import logging
        logger = logging.getLogger(__name__)
        logger.exception("Error retrieving documents: %s", e)
        return {"retrieved_docs": "[Could not retrieve documents — continuing without external docs]."}


def generate_answer(state):
    user_content = f"Kafka Knowledge:\n{state['retrieved_docs']}\n\nUser Question:\n{state['user_query']}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    resp = openai_client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=OPENAI_TEMPERATURE,
    )

    # Extract text from response
    text = ""
    if resp and getattr(resp, "choices", None):
        choice = resp.choices[0]
        if getattr(choice, "message", None) and getattr(choice.message, "content", None):
            text = choice.message.content
        elif getattr(choice, "text", None):
            text = choice.text

    return {"final_answer": text}
