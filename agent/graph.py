from langgraph.graph import StateGraph
from agent.state import AgentState
from agent.nodes import retrieve_docs, generate_answer

graph = StateGraph(AgentState)

graph.add_node("retrieve", retrieve_docs)
graph.add_node("generate", generate_answer)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.set_finish_point("generate")

app = graph.compile()

def run_agent(query: str):
    result = app.invoke({"user_query": query})
    return result["final_answer"]
