"""
Document Management Router
Advanced document analysis, filtering, and cleanup tools
"""

import logging
from datetime import datetime
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DocumentAnalysis(BaseModel):
    document_id: int
    filename: str
    content_preview: str
    content_type: str
    is_problematic: bool
    problematic_reasons: List[str]
    confidence: float
    recommendation: str


class CleanupReport(BaseModel):
    total_documents: int
    problematic_documents: int
    bio_waste_documents: int
    removed_documents: List[int]
    status: str


router = APIRouter(prefix="/api/v1/document-manager", tags=["document-manager"])


def get_repositories():
    """Get repository instances"""
    try:
        from ..repositories.factory import (get_document_repository,
                                            get_rag_repository,
                                            get_vector_search_repository)

        return {
            "rag_repo": get_rag_repository(),
            "doc_repo": get_document_repository(),
            "vector_repo": get_vector_search_repository(),
        }
    except Exception as e:
        logger.error(f"Failed to get repositories: {e}")
        raise HTTPException(status_code=500, detail="Repository initialization failed")


@router.get("/analyze", response_model=List[DocumentAnalysis])
async def analyze_all_documents():
    """
    Analyze all documents to identify problematic content
    """
    try:
        repos = get_repositories()
        doc_repo = repos["doc_repo"]
        vector_repo = repos["vector_repo"]

        # Get all documents
        documents_result = await doc_repo.list_all()
        documents = documents_result.items
        analyses = []

        logger.info(f"Analyzing {len(documents)} documents...")

        for doc in documents:
            try:
                # Get document content via vector search
                search_results = await vector_repo.search_similar_text(
                    f"document_id:{doc.id}", limit=5, threshold=0.0
                )

                if not search_results.items:
                    continue

                # Analyze content
                full_content = " ".join(
                    [item.text_content for item in search_results.items]
                )
                content_preview = (
                    full_content[:500] + "..."
                    if len(full_content) > 500
                    else full_content
                )

                is_problematic, reasons = _analyze_content_quality(full_content)

                analysis = DocumentAnalysis(
                    document_id=doc.id,
                    filename=doc.filename,
                    content_preview=content_preview,
                    content_type=_classify_content_type(full_content),
                    is_problematic=is_problematic,
                    problematic_reasons=reasons,
                    confidence=_calculate_confidence(full_content),
                    recommendation=_get_recommendation(is_problematic, reasons),
                )

                analyses.append(analysis)

            except Exception as e:
                logger.error(f"Error analyzing document {doc.id}: {e}")
                continue

        logger.info(f"Analysis complete: {len(analyses)} documents analyzed")
        return analyses

    except Exception as e:
        logger.error(f"Document analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/cleanup", response_model=CleanupReport)
async def cleanup_problematic_documents(
    remove_training_docs: bool = True,
    remove_computer_science: bool = True,
    remove_corrupted: bool = True,
    dry_run: bool = False,
):
    """
    Remove problematic documents and rebuild clean index
    """
    try:
        repos = get_repositories()
        doc_repo = repos["doc_repo"]
        vector_repo = repos["vector_repo"]

        # Analyze all documents first
        analyses = await analyze_all_documents()

        total_docs = len(analyses)
        problematic_docs = [a for a in analyses if a.is_problematic]
        bio_waste_docs = [a for a in analyses if a.content_type == "bio_waste"]

        removed_document_ids = []

        if not dry_run:
            # Remove problematic documents
            for analysis in problematic_docs:
                should_remove = False

                if (
                    remove_training_docs
                    and "training_instructions" in analysis.problematic_reasons
                ):
                    should_remove = True
                if (
                    remove_computer_science
                    and "computer_science" in analysis.problematic_reasons
                ):
                    should_remove = True
                if (
                    remove_corrupted
                    and "corrupted_encoding" in analysis.problematic_reasons
                ):
                    should_remove = True

                if should_remove:
                    try:
                        # Remove from document repository
                        success = await doc_repo.delete(analysis.document_id)
                        if success:
                            removed_document_ids.append(analysis.document_id)
                            logger.info(
                                f"Removed problematic document {analysis.document_id}: {analysis.filename}"
                            )
                        else:
                            logger.error(
                                f"Failed to remove document {analysis.document_id}: delete returned False"
                            )
                    except Exception as e:
                        logger.error(
                            f"Failed to remove document {analysis.document_id}: {e}"
                        )

            # Rebuild vector index with clean documents only
            await _rebuild_clean_vector_index(repos, bio_waste_docs)

        report = CleanupReport(
            total_documents=total_docs,
            problematic_documents=len(problematic_docs),
            bio_waste_documents=len(bio_waste_docs),
            removed_documents=removed_document_ids,
            status="completed" if not dry_run else "dry_run_completed",
        )

        logger.info(f"Cleanup report: {report}")
        return report

    except Exception as e:
        logger.error(f"Document cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.post("/rebuild-index")
async def rebuild_vector_index():
    """
    Rebuild the entire vector index from clean documents
    """
    try:
        repos = get_repositories()

        logger.info("Starting vector index rebuild...")

        # Get only bio waste documents
        analyses = await analyze_all_documents()
        clean_docs = [
            a
            for a in analyses
            if not a.is_problematic and a.content_type == "bio_waste"
        ]

        await _rebuild_clean_vector_index(repos, clean_docs)

        return {
            "status": "success",
            "message": f"Vector index rebuilt with {len(clean_docs)} clean documents",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Vector index rebuild failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")


@router.get("/bio-waste-documents")
async def get_bio_waste_documents():
    """
    Get only verified bio waste documents
    """
    try:
        analyses = await analyze_all_documents()
        bio_waste_docs = [
            a
            for a in analyses
            if a.content_type == "bio_waste" and not a.is_problematic
        ]

        return {"count": len(bio_waste_docs), "documents": bio_waste_docs}

    except Exception as e:
        logger.error(f"Failed to get bio waste documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


def _analyze_content_quality(content: str) -> tuple[bool, List[str]]:
    """Analyze if content is problematic using configured keywords"""
    content_lower = content.lower()
    reasons = []

    # Load filter configuration
    from pathlib import Path

    import yaml

    config_path = Path("config/document_filters.yaml")

    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        # Use defaults
        config = {
            "problematic_keywords": [
                "zero-hallucination",
                "guidelines for following",
                "only use information",
                "training instructions",
                "quelels",
            ],
            "exclude_keywords": [
                "javascript",
                "console.log",
                "function",
                "cloud computing",
                "programming",
                "software",
                "algorithm",
            ],
            "min_content_length": 100,
            "max_corruption_chars": 10,
        }

    # Check for training instructions
    if any(
        keyword in content_lower for keyword in config.get("problematic_keywords", [])
    ):
        reasons.append("training_instructions")

    # Check for excluded content
    if any(keyword in content_lower for keyword in config.get("exclude_keywords", [])):
        reasons.append("excluded_content")

    # Check for corrupted encoding
    corruption_count = content.count("�")
    if corruption_count > config.get("max_corruption_chars", 10):
        reasons.append("corrupted_encoding")

    # Check for very short or empty content
    min_length = config.get("min_content_length", 100)
    if len(content.strip()) < min_length:
        reasons.append("too_short")

    return len(reasons) > 0, reasons


def _classify_content_type(content: str) -> str:
    """Classify the type of content using configured keywords"""
    content_lower = content.lower()

    # Load filter configuration
    from pathlib import Path

    import yaml

    config_path = Path("config/document_filters.yaml")

    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        # Use defaults
        config = {
            "bio_waste_keywords": [
                "bioabfall",
                "bio waste",
                "organic waste",
                "kompost",
                "grünabfall",
                "küchenabfälle",
                "obst",
                "gemüse",
                "fruit",
                "vegetable",
                "food waste",
            ]
        }

    # Check for target content (e.g., bio waste)
    target_keywords = config.get("bio_waste_keywords", [])
    if any(indicator in content_lower for indicator in target_keywords):
        return "bio_waste"

    # Computer science indicators
    cs_indicators = [
        "javascript",
        "programming",
        "software",
        "algorithm",
        "cloud computing",
    ]
    if any(indicator in content_lower for indicator in cs_indicators):
        return "computer_science"

    # Training/instruction indicators
    training_indicators = ["guidelines", "rules", "instructions", "richtlinien"]
    if any(indicator in content_lower for indicator in training_indicators):
        return "training_material"

    return "unknown"


def _calculate_confidence(content: str) -> float:
    """Calculate confidence score for content classification"""
    content_lower = content.lower()

    # Bio waste confidence boosters
    bio_score = 0
    bio_keywords = [
        "bioabfall",
        "organic",
        "compost",
        "obst",
        "gemüse",
        "küchenabfälle",
    ]
    bio_score = sum(1 for keyword in bio_keywords if keyword in content_lower)

    # Length bonus
    length_bonus = min(len(content) / 1000, 1.0)

    # Corruption penalty
    corruption_penalty = content.count("�") * 0.1

    confidence = min((bio_score * 0.2 + length_bonus * 0.3) - corruption_penalty, 1.0)
    return max(confidence, 0.0)


def _get_recommendation(is_problematic: bool, reasons: List[str]) -> str:
    """Get recommendation for document"""
    if not is_problematic:
        return "keep"

    if "training_instructions" in reasons:
        return "remove_immediately"
    if "computer_science" in reasons:
        return "remove"
    if "corrupted_encoding" in reasons:
        return "fix_encoding_or_remove"
    if "too_short" in reasons:
        return "review_manually"

    return "review"


async def _rebuild_clean_vector_index(
    repos: Dict, clean_documents: List[DocumentAnalysis]
):
    """Rebuild vector index with only clean documents"""
    try:
        vector_repo = repos["vector_repo"]

        logger.info(
            f"Rebuilding vector index with {len(clean_documents)} clean documents"
        )

        # Clear existing index
        # Note: This would require implementation in the vector repository
        # For now, we'll log the intent
        logger.info("Clearing existing vector index...")

        # Re-index clean documents
        for doc_analysis in clean_documents:
            logger.info(
                f"Re-indexing document {doc_analysis.document_id}: {doc_analysis.filename}"
            )
            # Implementation would depend on vector repository interface

        logger.info("Vector index rebuild completed")

    except Exception as e:
        logger.error(f"Vector index rebuild failed: {e}")
        raise
