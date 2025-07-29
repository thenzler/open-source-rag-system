#!/usr/bin/env python3
"""
Main entry point for the RAG system
"""

import sys
import os

# Add src to path if it exists
if os.path.exists("src"):
    sys.path.insert(0, "src")

# Try to import from different locations
try:
    from core.main import main
except ImportError:
    try:
        from src.main import main
    except ImportError:
        print("Error: Could not find main application module")
        print("Please ensure the application is properly installed")
        sys.exit(1)

if __name__ == "__main__":
    main()
