#!/usr/bin/env python3
"""Debug LLM answer generation"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ollama_client import get_ollama_client
    
    print("=== Testing LLM Answer Generation ===")
    
    # Get client
    client = get_ollama_client()
    
    # Test availability
    print(f"1. Ollama available: {client.is_available()}")
    
    # Test simple generation
    simple_prompt = "Sage 'Hallo' auf Deutsch."
    simple_answer = client.generate_answer(simple_prompt, max_tokens=50, temperature=0.1)
    print(f"2. Simple test: '{simple_answer}'")
    
    # Test with context
    context = """
    Bioabfall Sammlung:
    - Obst- und Gemüsereste
    - Kaffeesatz und Teebeutel  
    - Eierschalen
    - Gartenabfälle
    - Blumen
    """
    
    query = "Was gehört in den Biomüll?"
    
    # Use the new intelligent prompt
    prompt = f"""Du bist ein intelligenter KI-Assistent, der Fragen basierend auf bereitgestellten Dokumenten beantwortet.

**DEINE AUFGABE - Schritt für Schritt:**

1. **ANALYSIERE DIE FRAGE:** 
   - Was genau möchte der Nutzer wissen?
   - Welche Art von Information wird gesucht? (Anleitung, Definition, Ort, Zeit, etc.)

2. **DURCHSUCHE DEN KONTEXT INTELLIGENT:**
   - Lies ALLE bereitgestellten Dokumentabschnitte sorgfältig
   - Identifiziere relevante Informationen für JEDEN Teil der Frage

3. **ANTWORTE INTELLIGENT:**
   ✓ Beantworte die Frage VOLLSTÄNDIG und PRÄZISE
   ✓ Strukturiere deine Antwort logisch (Hauptpunkte → Details)
   ✓ Verwende konkrete Informationen aus den Dokumenten

**Kontextdokumente:**
{context}

**Benutzeranfrage:**
{query}

**Deine intelligente Antwort:**"""
    
    print(f"3. Testing with context...")
    print(f"   Prompt length: {len(prompt)} characters")
    
    answer = client.generate_answer(prompt, max_tokens=200, temperature=0.3)
    print(f"4. LLM Answer: '{answer}'")
    print(f"   Answer length: {len(answer) if answer else 0}")
    
    if not answer:
        print("❌ LLM returned empty answer!")
        
        # Test health check
        health = client.health_check()
        print("5. Health check:")
        for key, value in health.items():
            print(f"   {key}: {value}")
    else:
        print("✅ LLM working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()