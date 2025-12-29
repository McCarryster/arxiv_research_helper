"""
Dependency injection module for manager instances.

Provides singleton factory functions for backend manager instances with
support for path changes and testing scenarios.
"""

import app.config as config
from typing import Optional
from backend.pdf_manager import ArxivPDFManager
from typing import Optional, Dict, Any


# Global singleton cache variables
_arxiv_pdf_manager: Optional[ArxivPDFManager] = None


def get_arxiv_pdf_manager(recreate: bool = False) -> ArxivPDFManager:
    """
    Get or create the singleton ArxivPDFManager instance configured for PostgreSQL.
    
    Args:
        recreate: Force creation of a new instance even if one exists.
        show_progress: Whether to display tqdm progress bars during manager operations.
        
    Returns:
        ArxivPDFManager instance.
    """
    global _arxiv_pdf_manager
    
    # Only create if it doesn't exist OR if the path has changed (important for tests!)
    current_pg_config: Dict[str, Any] = config.PG
    
    # if _arxiv_pdf_manager is None or recreate:
    if _arxiv_pdf_manager is None or _arxiv_pdf_manager.db_config != current_pg_config or recreate:
        _arxiv_pdf_manager = ArxivPDFManager(current_pg_config)
    
    return _arxiv_pdf_manager