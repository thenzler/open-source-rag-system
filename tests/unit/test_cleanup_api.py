#!/usr/bin/env python3
"""
Test the document cleanup API directly
"""
import asyncio
import sys
sys.path.append('.')

async def test_cleanup():
    """Test the document cleanup functionality directly"""
    try:
        from core.routers.document_manager import analyze_all_documents, cleanup_problematic_documents
        
        print("Testing document analysis...")
        
        # Analyze documents
        analyses = await analyze_all_documents()
        print(f"Found {len(analyses)} documents")
        
        bio_waste_count = sum(1 for a in analyses if a.content_type == "bio_waste" and not a.is_problematic)
        problematic_count = sum(1 for a in analyses if a.is_problematic)
        
        print(f"Bio waste documents: {bio_waste_count}")
        print(f"Problematic documents: {problematic_count}")
        
        if problematic_count > 0:
            print(f"\nCleaning up {problematic_count} problematic documents...")
            
            # Cleanup problematic documents
            report = await cleanup_problematic_documents(
                remove_training_docs=True,
                remove_computer_science=True,
                remove_corrupted=True,
                dry_run=False
            )
            
            print(f"Cleanup report: {report}")
            print("✓ Cleanup completed successfully!")
        else:
            print("✓ No problematic documents found - system is clean!")
            
        return True
        
    except Exception as e:
        print(f"✗ Cleanup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_cleanup())
    if success:
        print("✓ Test completed successfully!")
        sys.exit(0)
    else:
        print("✗ Test failed!")
        sys.exit(1)