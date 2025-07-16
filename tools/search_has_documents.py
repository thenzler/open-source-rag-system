#!/usr/bin/env python3
"""Search for has_documents function in simple_api.py"""

with open("C:/Users/THE/open-source-rag-system/simple_api.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "def has_documents" in line:
        print(f"Found has_documents function at line {i+1}:")
        # Print the function and several lines after it
        for j in range(i, min(i+20, len(lines))):
            print(f"{j+1:4d}: {lines[j].rstrip()}")
        break
else:
    print("has_documents function not found!")
    
    # Search for where it's called to understand expected behavior
    print("\nLooking for calls to has_documents()...")
    for i, line in enumerate(lines):
        if "has_documents()" in line:
            print(f"Called at line {i+1}: {lines[i].strip()}")