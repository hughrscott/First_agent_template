from langgraph.graph import StateGraph, END
from typing import TypedDict # For AgentState

# Import your state and nodes from the nodes.py file
from .nodes import (
    AgentState, # The TypedDict for your agent's state
    MediaRouter,
    TextExtractionNode,
    ImageExtractionNode,
    AudioExtractionNode,
    DataExtractionNode,
    VideoExtractionNode,
    AnswerRefinementNode,
    WebSearchNode,
)

# Workflow Assembly (paste the code here)
# Define the LangGraph workflow
# Workflow Assembly
workflow = StateGraph(AgentState)
nodes = [
    "TextExtractionNode",
    "ImageExtractionNode",
    "AudioExtractionNode",
    "DataExtractionNode",
    "VideoExtractionNode",
    "WebSearchNode",
]

workflow.add_node("MediaRouter", MediaRouter)
for node in nodes:
    workflow.add_node(node, globals()[node])

# Add the refinement node
workflow.add_node("AnswerRefinementNode", AnswerRefinementNode)

workflow.set_conditional_entry_point(MediaRouter, {node: node for node in nodes})

for node in nodes:
    workflow.add_edge(node, "AnswerRefinementNode")

# The refinement node then goes to END
workflow.add_edge("AnswerRefinementNode", END)

app = workflow.compile()
