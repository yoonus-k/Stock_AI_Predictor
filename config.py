import sys
from pathlib import Path

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Add to Python path if not already there
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Function to get paths relative to project root
def get_project_path(relative_path):
    return PROJECT_ROOT / relative_path


# usage example
# In any file that needs path management
# from config import PROJECT_ROOT