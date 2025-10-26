# setup_env.py
import sys
import os

# Get the absolute path to the project root (this file's directory)
project_root = os.path.dirname(os.path.abspath(__file__))

# Add the root to sys.path so Python can locate all project modules
if project_root not in sys.path:
    sys.path.append(project_root)


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.abspath(__file__))
    print(project_root)