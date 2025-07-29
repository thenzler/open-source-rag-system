#!/usr/bin/env python3
"""
Fix Unicode characters in test files for Windows compatibility
"""

import os
import re

def fix_unicode_in_file(filepath):
    """Fix unicode characters in a file"""
    print(f"Fixing unicode in {filepath}")
    
    # Read the file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define replacements
    replacements = {
        '✓': '+',
        '✗': 'X',
        '❌': 'X',
        '✅': '+',
        '⚠️': '!',
        '🔧': '',
        '🤖': '',
        '📄': '',
        '🔍': '',
        '📋': '',
        '🎉': '',
        '="🔧': '="',
        '="🤖': '="',
        '="📄': '="',
        '="🔍': '="',
        '="📋': '="',
        '="🎉': '="',
    }
    
    # Apply replacements
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {filepath}")

def main():
    """Fix unicode in all test files"""
    test_files = [
        'tests/test_ollama_integration.py',
        'tests/test_services.py'
    ]
    
    for filepath in test_files:
        if os.path.exists(filepath):
            fix_unicode_in_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    main()