import os
from openai import OpenAI
from rag.retriever import get_retriever
from prompts.kafka_prompt import SYSTEM_PROMPT

openai_client = OpenAI()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))


def retrieve_docs(state):
    retriever = get_retriever()
    docs = retriever.get_relevant_documents(state["user_query"])
    content = "\n".join(doc.page_content for doc in docs)
    return {"retrieved_docs": content}


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
