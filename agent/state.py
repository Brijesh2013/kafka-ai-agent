from typing import TypedDict

class AgentState(TypedDict):
    user_query: str
    retrieved_docs: str
    final_answer: str
