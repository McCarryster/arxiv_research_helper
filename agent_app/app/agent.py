import os
import app.config as config, prompts
from typing import List, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
    AIMessage
)
from langchain_anthropic import ChatAnthropic
from app.tools.pdf_tools import get_arxiv_pdf_content

# LangSmith integration for tracing
try:
    from langsmith import traceable
    from langsmith.wrappers import wrap_anthropic
    LANGSMITH_AVAILABLE = True
    
    # Configure LangSmith if API key is available
    if config.LANGSMITH_API_KEY:
        os.environ["LANGSMITH_API_KEY"] = config.LANGSMITH_API_KEY
        os.environ["LANGSMITH_TRACING"] = "true"
        if config.LANGSMITH_PROJECT:
            os.environ["LANGSMITH_PROJECT"] = config.LANGSMITH_PROJECT
except ImportError:
    LANGSMITH_AVAILABLE = False
    # Create a no-op decorator if LangSmith is not available
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    def wrap_anthropic(client):
        return client

class ArxivAgent:
    """
    Arxiv agent class for LLM-powered task execution with tool calling.
    """
    
    def __init__(
        self,
        name: str,
        llm: ChatAnthropic,
        tools: List[BaseTool],
        system_prompt: str = "You are a helpful research assistant. Use the tools provided to find and read Arxiv papers.",
        steps_limit: int = 5
    ) -> None:
        self.name: str = name
        # Bind tools to the LLM so it knows what it can call
        self.llm_with_tools = llm.bind_tools(tools)
        self.tools_map: Dict[str, BaseTool] = {tool.name: tool for tool in tools}
        self.system_prompt: str = system_prompt
        self.steps_limit: int = steps_limit

    @traceable
    def run(self, query: str) -> str:
        """Main execution loop for the agent."""
        messages: List[BaseMessage] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=query)
        ]

        for _ in range(self.steps_limit):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)

            # If the response is just text and no tool calls, we are done
            if not isinstance(response, AIMessage) or not response.tool_calls:
                return str(response.content)

            # Process tool calls
            for tool_call in response.tool_calls:
                tool_name: str = tool_call["name"].lower()
                tool_args: Dict[str, Any] = tool_call["args"]
                
                if tool_name in self.tools_map:
                    tool_output = self.tools_map[tool_name].invoke(tool_args)
                    messages.append(
                        ToolMessage(
                            content=str(tool_output),
                            tool_call_id=tool_call["id"]
                        )
                    )
                else:
                    messages.append(
                        ToolMessage(
                            content=f"Error: Tool '{tool_name}' not found.",
                            tool_call_id=tool_call["id"]
                        )
                    )

        return "Error: Reached maximum steps limit without a final answer."

# --- Initialization ---
def initialize_agent() -> ArxivAgent:
    llm = ChatAnthropic(
        model_name=config.DEFAULT_MODEL,
        max_tokens_to_sample=config.MAX_TOKENS_PER_RESPONSE,
        temperature=config.TEMPERATURE,
        timeout=None,
        stop=None,
    )
    
    if LANGSMITH_AVAILABLE:
        llm = wrap_anthropic(llm)

    # Define the toolset
    tools: List[BaseTool] = [get_arxiv_pdf_content]

    system_prompt = prompts.ARXIV_AGENT_SYSTEM_PROMPT

    # Create the agent instance
    return ArxivAgent(
        name="ArxivResearcher",
        llm=llm,
        tools=tools,
        steps_limit=config.STEPS_LIMIT_AGENT,
        system_prompt=system_prompt,
    )