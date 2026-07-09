"""Agentic answer loop: planner -> retrieve -> synthesize -> critic (ADR-007).

LangGraph orchestrates the loop; every LLM call goes through the existing
``LLMProvider`` protocol (ADR-001), never LangChain model wrappers. Contracts
live in ``state``; the compiled graph and the ``AgenticPipeline`` facade live
in ``graph``.
"""
