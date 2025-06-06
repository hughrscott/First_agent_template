# Clean, simple agent.py - let the LLM choose
from langgraph.graph import StateGraph, END
from typing import TypedDict

from agent.nodes import (
    AgentState,
    SmartRouter,  # Our new simple LLM-driven router
    # Keep your existing working nodes
    CalculatorNode,
    WebSearchNode,
    DataExtractionNode,
    ImageExtractionNode,
    AudioExtractionNode,
    VideoExtractionNode,
    MultiStepNode,
    AnswerRefinementNode,
)

# Simple workflow - let the LLM decide everything
workflow = StateGraph(AgentState)

# Available execution nodes
execution_nodes = [
    "CalculatorNode",
    "WebSearchNode", 
    "DataExtractionNode",
    "ImageExtractionNode",
    "AudioExtractionNode",
    "VideoExtractionNode", 
    "MultiStepNode",
]

# Add the smart router
workflow.add_node("SmartRouter", SmartRouter)

# Add all execution nodes
for node in execution_nodes:
    workflow.add_node(node, globals()[node])

# Add refinement
workflow.add_node("AnswerRefinementNode", AnswerRefinementNode)

# Simple flow: Router -> Execution -> Refinement -> Done
workflow.set_conditional_entry_point(SmartRouter, {node: node for node in execution_nodes})

# All execution nodes go to refinement
for node in execution_nodes:
    workflow.add_edge(node, "AnswerRefinementNode")

workflow.add_edge("AnswerRefinementNode", END)

app = workflow.compile()