# Zero Hallucination Tolerance Implementation Plan

## 🎯 **Core Goal**
Implement strict controls to ensure the RAG system NEVER hallucinates and ONLY provides information that exists in the document database with precise source citations.

## 📋 **Implementation Steps**

### **Phase 1: Core Guardrails (High Priority)**

#### **1. Similarity Threshold Enforcement**
- **File**: `core/services/query_service.py`
- **Action**: Reject answers when vector similarity < 0.8
- **Implementation**:
  ```python
  if max_similarity < 0.8:
      return "Dazu finde ich keine ausreichend relevanten Informationen in den verfügbaren Dokumenten."
  ```

#### **2. Mandatory Source Citations**
- **File**: `core/ollama_client.py` - `_create_rag_prompt()`
- **Action**: Every response MUST include document ID and chunk reference
- **Format**: 
  ```
  Antwort: [Inhalt]
  Quelle: Dokument ID [X], Abschnitt [Y]
  Download: /api/v1/documents/[X]/download
  ```

#### **3. "I Don't Know" Logic**
- **File**: `core/services/query_service.py`
- **Action**: Explicit refusal when no relevant docs found
- **Triggers**:
  - No search results
  - All results below similarity threshold
  - Query asks for external knowledge

#### **4. Enhanced System Prompt**
- **File**: `data/training_data/arlesheim_german/Modelfile_final`
- **Action**: Add strict source citation requirements
- **New Rules**:
  ```
  STRIKT: Antworte NUR wenn:
  1. Information EXPLIZIT in Dokumenten vorhanden
  2. Genaue Quelle benennbar (Dokument-ID + Abschnitt)
  3. Sonst: "Keine Informationen in verfügbaren Dokumenten"
  ```

### **Phase 2: Response Validation (Medium Priority)**

#### **5. Content Verification**
- **File**: `core/services/validation_service.py` (new)
- **Action**: Verify LLM response content matches source chunks
- **Method**: Text similarity check between answer and source

#### **6. Query Preprocessing**
- **File**: `core/services/query_service.py`
- **Action**: Filter queries requiring external knowledge
- **Examples to reject**: "What's the weather?", "Who is the president?"

#### **7. Confidence Scoring**
- **File**: `core/services/query_service.py`
- **Action**: Implement multi-factor confidence scoring
- **Factors**: 
  - Vector similarity scores
  - Number of matching documents
  - Consensus across chunks

### **Phase 3: Monitoring & Transparency (Low Priority)**

#### **8. Response Audit Logging**
- **File**: `core/repositories/audit_repository.py`
- **Action**: Log all responses with source references for validation
- **Data**: Query, response, source chunks, confidence scores

#### **9. Frontend Transparency**
- **File**: `static/index.html`
- **Action**: Show confidence indicators and source links
- **UI Elements**:
  - Confidence meter
  - "Sources used" section
  - Direct download links

## 🔧 **Technical Implementation Details**

### **Query Processing Flow**
```
1. User Query → Preprocessing (filter external knowledge)
2. Vector Search → Similarity check (>0.8 threshold)
3. LLM Generation → With strict source prompt
4. Response Validation → Content matches sources
5. Audit Logging → Track sources and confidence
6. User Response → With citations and download links
```

### **Response Format**
```json
{
  "answer": "Die Öffnungszeiten sind Mo-Fr 08:00-17:00",
  "confidence": 0.95,
  "sources": [
    {
      "document_id": 42,
      "chunk_id": 123,
      "similarity": 0.92,
      "download_url": "/api/v1/documents/42/download"
    }
  ],
  "has_sufficient_confidence": true
}
```

### **Error Responses**
```json
{
  "answer": "Dazu finde ich keine Informationen in den verfügbaren Dokumenten.",
  "confidence": 0.0,
  "sources": [],
  "has_sufficient_confidence": false,
  "reason": "no_relevant_documents"
}
```

## 🚨 **Critical Success Criteria**

### **Zero Tolerance Metrics**
- **No Hallucinations**: 0% responses with information not in documents
- **Source Citation**: 100% answers include precise document references
- **Appropriate Refusal**: System says "I don't know" when confidence < threshold
- **Transparency**: Users can verify every answer against source documents

### **Testing Strategy**
1. **Known Information**: Test with questions answerable from documents
2. **External Knowledge**: Test with questions requiring outside information
3. **Partial Information**: Test with questions partially answerable
4. **Ambiguous Questions**: Test edge cases and unclear queries

## 📝 **Configuration Files to Update**

### **Model Configuration**
- `data/training_data/arlesheim_german/Modelfile_final`
- `config/llm_config.yaml`

### **System Configuration**
- `core/services/query_service.py` 
- `core/ollama_client.py`
- `core/repositories/audit_repository.py`

### **Frontend Updates**
- `static/index.html`
- Add confidence indicators and source transparency

## 🎯 **Success Validation**

Test queries that should trigger refusal:
- "What's the weather today?"
- "Who won the latest football match?"
- "What's the capital of France?"

Test queries that should work with citations:
- "Wann ist die Gemeindeverwaltung geöffnet?"
- "Wie entsorge ich Grünabfall?"
- "Wo finde ich Informationen zu Baugesuchen?"

---

**Implementation Priority**: Start with Phase 1 (high priority items) to establish core zero-hallucination foundation.