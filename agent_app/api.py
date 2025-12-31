from app.agent import ArxivAgent, initialize_agent
from typing import Final
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AGENT_INSTANCE: Final[ArxivAgent] = initialize_agent()

def process_query(query: str) -> str:
    """
    Process natural language queries about scientific papers.
    If a response to this question cannot be given with the tools available,
    return "<NO_ANSWER_POSSIBLE>".

    Args:
        query: Natural language query string

    Returns:
        Response to the query or a fallback token.
    """
    # 1. Validation for empty queries
    if not query.strip():
        return "<NO_ANSWER_POSSIBLE>"

    try:
        # 2. Run the agent logic
        response = AGENT_INSTANCE.run(query=query)
        
        # 3. Check for empty strings or specific error markers from the agent
        if not response or "Error: Reached maximum steps limit" in response:
            return "<NO_ANSWER_POSSIBLE>"
            
        return response

    except Exception as e:
        return "<NO_ANSWER_POSSIBLE>"

def main():
    """CLI interface for testing API methods."""
    parser = argparse.ArgumentParser(
        description="Property Management System API"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--query",
        type=str,
        help="Process a natural language query"
    )

    args = parser.parse_args()

    logger.info(f"Processing query: {args.query}")
    result = process_query(args.query)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
