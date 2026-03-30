import os
import sys


def ensure_project_root() -> str:
    """
    Ensure the project root is importable when running subdirectory scripts directly.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root
