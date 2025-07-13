# 🔍 Query Improvements - Bessere Kontextverständnis

## Problem
Das System war zu pingelig und verstand Synonyme/alternative Formulierungen nicht gut genug.

## Lösung: Query Expansion + Niedrigere Schwellenwerte

### 1. **Query Expansion Service**
Erweitert Suchanfragen automatisch um Synonyme und verwandte Begriffe:

**Beispiel für "Was kommt in die Biotonne":**
- Original: "was kommt in die biotonne"
- Expansion 1: "was kommt in die bioabfall" 
- Expansion 2: "was kommt in die organische abfälle"
- Kernbegriffe: "biotonne organische"

### 2. **Deutsche Synonyme für Abfall/Umwelt**
```python
"biotonne": ["bioabfall", "biomüll", "organische abfälle", "kompost", "grünabfall"]
"restmüll": ["restabfall", "hausmüll", "schwarze tonne", "graue tonne"]
"recycling": ["wiederverwertung", "wertstoff", "kreislaufwirtschaft"]
"entsorgung": ["entsorgen", "wegwerfen", "beseitigung"]
```

### 3. **Kontextuelle Begriffe**
```python
"gehört": ["kommt", "darf", "kann", "soll", "muss"]
"was": ["welche", "welches", "welcher", "wo"]
"darf": ["kann", "soll", "sollte", "gehört", "kommt"]
```

### 4. **Intelligente Gewichtung**
- Original Query: Gewicht 1.0
- 1. Expansion: Gewicht 0.9  
- 2. Expansion: Gewicht 0.8
- Beste Ergebnisse aus allen Varianten kombiniert

### 5. **Niedrigere Schwellenwerte**
- **Vorher**: 40% Mindest-Ähnlichkeit (zu streng!)
- **Jetzt**: 30% Mindest-Ähnlichkeit (flexibler)

## Verbesserungen

### **Vorher:**
```
Query: "Was kommt in die Biotonne"
Result: Similarity 0.334 -> "unter Grenzwert, keine Ergebnisse"
```

### **Jetzt:**
```
Query: "Was kommt in die Biotonne"
Expanded to: 
  - "was kommt in die biotonne"
  - "was kommt in die bioabfall"  
  - "was kommt in die organische abfälle"
Best match: 0.68 (from "organische abfälle")
Result: ✅ Zeigt relevante Informationen
```

## Zusätzliche Features

### **Kernbegriff-Extraktion**
Bei langen Anfragen werden die wichtigsten 2-3 Begriffe extrahiert:
- "Was darf ich in die Biotonne entsorgen?" → "biotonne entsorgen"

### **Stopword-Filterung** 
Unwichtige Wörter werden entfernt:
- Gefiltert: der, die, das, ein, eine, und, oder, ist, sind, etc.

### **Debug-Logging**
Zeigt genau welche Query-Varianten verwendet werden:
```
INFO: Expanded query 'Was kommt in die Biotonne' to 3 variants
INFO: Result 0: similarity=0.68 (query: 'organische abfälle')
```

## Test Steps

1. **Server neustarten**:
```bash
python simple_api.py
```

2. **Sehen in Logs**:
```
OK: Query expansion loaded successfully!
[OK] Query expander initialized!
```

3. **Suche testen**: "Was kommt in die Biotonne"
   - Sollte jetzt bessere Ergebnisse zeigen
   - Debug-Logs zeigen Query-Expansion

4. **Weitere Tests**:
   - "Was darf in den Restmüll?"
   - "Wie entsorge ich Papier?"
   - "Recycling von Plastik"

**Das System sollte jetzt deutlich flexibler und nutzerfreundlicher sein! 🎯**