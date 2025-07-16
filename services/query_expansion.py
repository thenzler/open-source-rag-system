#!/usr/bin/env python3
"""
Query Expansion Service
Erweitert Suchanfragen um Synonyme und verwandte Begriffe
"""

import logging
from typing import List, Dict, Set
import re

logger = logging.getLogger(__name__)

class QueryExpander:
    """Erweitert Suchanfragen um Synonyme und verwandte Begriffe"""
    
    def __init__(self):
        # Deutsche Synonyme für häufige Abfall-/Umwelt-Begriffe
        self.synonyms = {
            "biotonne": ["bioabfall", "biomüll", "organische abfälle", "kompost", "grünabfall"],
            "restmüll": ["restabfall", "hausmüll", "schwarze tonne", "graue tonne"],
            "recycling": ["wiederverwertung", "wertstoff", "kreislaufwirtschaft"],
            "energie": ["strom", "elektrizität", "power", "energieverbrauch"],
            "abfall": ["müll", "waste", "entsorgung", "abfälle"],
            "umwelt": ["natur", "ökologie", "umweltschutz", "nachhaltigkeit"],
            "nachhaltig": ["ökologisch", "umweltfreundlich", "grün", "eco"],
            "entsorgung": ["entsorgen", "wegwerfen", "beseitigung", "müllentsorgung"],
            "kompost": ["kompostierung", "verrottung", "organisch", "biologisch abbaubar"],
            "papier": ["pappe", "karton", "zeitung", "altpapier"],
            "glas": ["glasflaschen", "glasbehälter", "glascontainer"],
            "metall": ["dosen", "alu", "aluminium", "blech", "eisen"],
            "plastik": ["kunststoff", "polymer", "plastic", "verpackung"]
        }
        
        # Kontextuelle Begriffe
        self.context_terms = {
            "gehört": ["kommt", "darf", "kann", "soll", "muss"],
            "tonne": ["container", "behälter", "sammlung", "entsorgung"],
            "was": ["welche", "welches", "welcher", "wo"],
            "darf": ["kann", "soll", "sollte", "gehört", "kommt"],
            "nicht": ["kein", "keine", "niemals", "verboten"]
        }
    
    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        Erweitert eine Suchanfrage um verwandte Begriffe
        
        Args:
            query: Original Suchanfrage
            max_expansions: Maximale Anzahl zusätzlicher Varianten
            
        Returns:
            Liste von Suchanfragen (Original + Varianten)
        """
        expanded_queries = [query]  # Original immer dabei
        query_lower = query.lower()
        
        try:
            # 1. Direkte Synonymersetzung
            for main_term, synonyms in self.synonyms.items():
                if main_term in query_lower:
                    for synonym in synonyms[:2]:  # Nur die besten 2 Synonyme
                        expanded_query = re.sub(
                            r'\b' + re.escape(main_term) + r'\b', 
                            synonym, 
                            query_lower, 
                            flags=re.IGNORECASE
                        )
                        if expanded_query != query_lower and expanded_query not in expanded_queries:
                            expanded_queries.append(expanded_query)
                            if len(expanded_queries) > max_expansions:
                                break
                
                # Auch umgekehrt prüfen (Synonym -> Hauptbegriff)
                for synonym in synonyms:
                    if synonym in query_lower:
                        expanded_query = re.sub(
                            r'\b' + re.escape(synonym) + r'\b', 
                            main_term, 
                            query_lower, 
                            flags=re.IGNORECASE
                        )
                        if expanded_query != query_lower and expanded_query not in expanded_queries:
                            expanded_queries.append(expanded_query)
                            if len(expanded_queries) > max_expansions:
                                break
            
            # 2. Kontextuelle Begriffe
            for main_term, alternatives in self.context_terms.items():
                if main_term in query_lower:
                    for alt in alternatives[:1]:  # Nur 1 Alternative pro Kontext
                        expanded_query = re.sub(
                            r'\b' + re.escape(main_term) + r'\b', 
                            alt, 
                            query_lower, 
                            flags=re.IGNORECASE
                        )
                        if expanded_query != query_lower and expanded_query not in expanded_queries:
                            expanded_queries.append(expanded_query)
                            if len(expanded_queries) > max_expansions:
                                break
            
            # 3. Kernbegriffe extrahieren (bei längeren Anfragen)
            words = query_lower.split()
            if len(words) > 3:
                # Wichtige Substantive finden
                important_words = []
                for word in words:
                    if len(word) > 4 and word not in ['kommt', 'gehört', 'kann', 'soll', 'muss', 'darf']:
                        important_words.append(word)
                
                if len(important_words) >= 2:
                    # Kurze Variante mit nur den wichtigsten Begriffen
                    short_query = ' '.join(important_words[:2])
                    if short_query not in expanded_queries:
                        expanded_queries.append(short_query)
            
            logger.info(f"Query expansion: '{query}' -> {len(expanded_queries)} variants")
            return expanded_queries[:max_expansions + 1]  # Original + max_expansions
            
        except Exception as e:
            logger.error(f"Error in query expansion: {e}")
            return [query]  # Fallback: nur original
    
    def get_core_terms(self, query: str) -> List[str]:
        """Extrahiert die wichtigsten Begriffe aus einer Anfrage"""
        words = query.lower().split()
        core_terms = []
        
        # Stopwords entfernen
        stopwords = {'der', 'die', 'das', 'ein', 'eine', 'und', 'oder', 'in', 'auf', 'mit', 'zu', 'für', 'von', 'bei', 'ist', 'sind', 'wird', 'werden'}
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) > 2 and clean_word not in stopwords:
                core_terms.append(clean_word)
        
        return core_terms

# Global instance
_query_expander = None

def get_query_expander() -> QueryExpander:
    """Get singleton instance"""
    global _query_expander
    if _query_expander is None:
        _query_expander = QueryExpander()
    return _query_expander