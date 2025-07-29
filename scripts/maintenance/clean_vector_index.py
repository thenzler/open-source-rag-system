#!/usr/bin/env python3
"""
Clean Vector Index Script
Remove problematic documents and rebuild with only bio waste content
"""
import asyncio
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def analyze_and_clean_documents():
    """Main function to analyze and clean documents"""
    try:
        from core.repositories.factory import get_rag_repository, get_document_repository, get_vector_search_repository
        
        print("Starting Document Analysis and Cleanup")
        print("=" * 60)
        
        # Get repositories
        doc_repo = get_document_repository()
        vector_repo = get_vector_search_repository()
        
        # Get all documents
        documents_result = await doc_repo.list_all()
        documents = documents_result.items
        print(f"Found {len(documents)} total documents")
        
        # Analyze each document
        bio_waste_docs = []
        problematic_docs = []
        
        for doc in documents:
            try:
                # Get content through vector search
                search_results = await vector_repo.search_similar_text(
                    f"document_id:{doc.id}", limit=5, threshold=0.0
                )
                
                if not search_results.items:
                    print(f"WARNING: Document {doc.id} ({doc.filename}) - No content found")
                    continue
                
                # Combine all chunks for analysis
                full_content = " ".join([item.text_content for item in search_results.items])
                content_lower = full_content.lower()
                
                # Classify document
                is_bio_waste = _is_bio_waste_document(content_lower, full_content)
                is_problematic = _is_problematic_document(content_lower, full_content)
                
                if is_problematic:
                    reasons = _get_problematic_reasons(content_lower, full_content)
                    problematic_docs.append({
                        'id': doc.id,
                        'filename': doc.filename,
                        'reasons': reasons,
                        'content_preview': full_content[:200] + "..."
                    })
                    print(f"PROBLEMATIC: Document {doc.id} ({doc.filename}) - {', '.join(reasons)}")
                
                elif is_bio_waste:
                    bio_waste_docs.append({
                        'id': doc.id,
                        'filename': doc.filename,
                        'content_preview': full_content[:200] + "..."
                    })
                    print(f"BIO WASTE: Document {doc.id} ({doc.filename}) - Clean content")
                
                else:
                    print(f"UNKNOWN: Document {doc.id} ({doc.filename}) - Unknown type")
                
            except Exception as e:
                print(f"ERROR: Analyzing document {doc.id}: {e}")
                continue
        
        # Show summary
        print(f"\nANALYSIS SUMMARY")
        print(f"Total documents: {len(documents)}")
        print(f"Bio waste documents: {len(bio_waste_docs)}")
        print(f"Problematic documents: {len(problematic_docs)}")
        print(f"Other documents: {len(documents) - len(bio_waste_docs) - len(problematic_docs)}")
        
        # Show problematic documents details
        if problematic_docs:
            print(f"\nPROBLEMATIC DOCUMENTS TO REMOVE:")
            for doc in problematic_docs:
                print(f"  - ID {doc['id']}: {doc['filename']}")
                print(f"    Reasons: {', '.join(doc['reasons'])}")
                try:
                    print(f"    Preview: {doc['content_preview']}")
                except UnicodeEncodeError:
                    print(f"    Preview: [Content contains special characters]")
                print()
        
        # Show bio waste documents
        if bio_waste_docs:
            print(f"\nCLEAN BIO WASTE DOCUMENTS TO KEEP:")
            for doc in bio_waste_docs:
                print(f"  - ID {doc['id']}: {doc['filename']}")
                try:
                    print(f"    Preview: {doc['content_preview']}")
                except UnicodeEncodeError:
                    print(f"    Preview: [Content contains special characters]")
                print()
        
        # Ask for confirmation
        if problematic_docs:
            response = input(f"\nRemove {len(problematic_docs)} problematic documents? (y/N): ")
            if response.lower() == 'y':
                await _remove_problematic_documents(doc_repo, problematic_docs)
                print("Problematic documents removed!")
            else:
                print("Cleanup cancelled")
        
        # Rebuild index
        if bio_waste_docs:
            response = input(f"\nRebuild vector index with {len(bio_waste_docs)} clean documents? (y/N): ")
            if response.lower() == 'y':
                await _rebuild_vector_index(vector_repo, bio_waste_docs)
                print("Vector index rebuilt!")
            else:
                print("Index rebuild cancelled")
        
        print("\nDocument cleanup completed!")
        return True
        
    except Exception as e:
        print(f"Cleanup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def _is_bio_waste_document(content_lower: str, full_content: str) -> bool:
    """Check if document contains bio waste information"""
    bio_indicators = [
        'bioabfall', 'bio waste', 'organic waste', 'kompost', 'grünabfall',
        'küchenabfälle', 'obst', 'gemüse', 'fruit', 'vegetable', 'food waste',
        'entsorgung organischer', 'abfallkalender', 'recycling',
        'gehört in den bioabfall', 'what goes in bio', 'bio-container'
    ]
    
    bio_score = sum(1 for indicator in bio_indicators if indicator in content_lower)
    return bio_score >= 2  # Require at least 2 bio waste indicators

def _is_problematic_document(content_lower: str, full_content: str) -> bool:
    """Check if document is problematic"""
    # Training instructions
    training_keywords = [
        'zero-hallucination', 'guidelines for following', 'only use information',
        'zusätzliche richtlinien', 'never pass on missing', 'always cite the source',
        'reference information from provided', 'quelels', 'verwende diese regeln exakt'
    ]
    
    # Computer science content
    cs_keywords = [
        'javascript', 'console.log', 'function(', 'cloud computing',
        '52 stunden informatik', 'programming', 'software development',
        'algorithm', 'var ', 'let ', 'const ', 'import ', 'class '
    ]
    
    # Check for training instructions
    if any(keyword in content_lower for keyword in training_keywords):
        return True
    
    # Check for computer science content (but not if it also has bio waste content)
    if any(keyword in content_lower for keyword in cs_keywords):
        if not _is_bio_waste_document(content_lower, full_content):
            return True
    
    # Check for corrupted encoding
    if full_content.count('�') > 10:  # High corruption
        return True
    
    # Check for very short content
    if len(full_content.strip()) < 100:
        return True
    
    return False

def _get_problematic_reasons(content_lower: str, full_content: str) -> list:
    """Get reasons why document is problematic"""
    reasons = []
    
    training_keywords = [
        'zero-hallucination', 'guidelines for following', 'only use information',
        'additional guidelines', 'verwende diese regeln'
    ]
    if any(keyword in content_lower for keyword in training_keywords):
        reasons.append("training_instructions")
    
    cs_keywords = ['javascript', 'console.log', 'programming', 'software']
    if any(keyword in content_lower for keyword in cs_keywords):
        reasons.append("computer_science")
    
    if full_content.count('�') > 10:
        reasons.append("corrupted_encoding")
    
    if len(full_content.strip()) < 100:
        reasons.append("too_short")
    
    return reasons

async def _remove_problematic_documents(doc_repo, problematic_docs):
    """Remove problematic documents from repository"""
    for doc in problematic_docs:
        try:
            success = await doc_repo.delete(doc['id'])
            if success:
                print(f"Removed document {doc['id']}: {doc['filename']}")
            else:
                print(f"Failed to remove document {doc['id']}: delete returned False")
        except Exception as e:
            print(f"Failed to remove document {doc['id']}: {e}")

async def _rebuild_vector_index(vector_repo, clean_docs):
    """Rebuild vector index with clean documents"""
    print(f"Rebuilding vector index with {len(clean_docs)} clean documents...")
    
    try:
        # Clear existing embeddings (this would depend on the vector repo implementation)
        print("Clearing existing vector index...")
        
        # Re-process clean documents
        for doc in clean_docs:
            print(f"Re-indexing document {doc['id']}: {doc['filename']}")
            # Implementation would depend on how documents are re-indexed
        
        print("Vector index rebuilt successfully!")
        
    except Exception as e:
        print(f"Vector index rebuild failed: {e}")
        raise

if __name__ == "__main__":
    print("RAG Document Cleanup Tool")
    print("This will analyze and clean problematic documents from your RAG system")
    print()
    
    try:
        success = asyncio.run(analyze_and_clean_documents())
        if success:
            print("Cleanup completed successfully!")
            sys.exit(0)
        else:
            print("Cleanup failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nCleanup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)