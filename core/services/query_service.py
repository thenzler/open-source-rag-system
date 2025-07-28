"""
Query Processing Service
Handles business logic for document querying and LLM interactions
"""
import logging
import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..repositories.interfaces import IDocumentRepository, IVectorSearchRepository
from ..repositories.audit_repository import SwissAuditRepository
from ..models.api_models import QueryRequest, SmartQueryResponse, DocumentChunk
try:
    from config.config import config
except ImportError:
    config = None

logger = logging.getLogger(__name__)

class QueryProcessingService:
    """Service for query processing business logic"""
    
    def __init__(
        self,
        doc_repo: IDocumentRepository,
        vector_repo: IVectorSearchRepository,
        audit_repo: SwissAuditRepository,
        ollama_client: Optional[Any] = None,
        confidence_config_path: Optional[str] = None
    ):
        self.doc_repo = doc_repo
        self.vector_repo = vector_repo
        self.audit_repo = audit_repo
        self.ollama_client = ollama_client
        
        # Initialize configurable confidence manager
        from .confidence_manager import ConfidenceManager
        self.confidence_manager = ConfidenceManager(confidence_config_path)
        
        # Query settings
        self.max_query_length = getattr(config, 'MAX_QUERY_LENGTH', 1000)
        self.default_limit = getattr(config, 'DEFAULT_SEARCH_LIMIT', 10)
        self.similarity_threshold = getattr(config, 'SIMILARITY_THRESHOLD', 0.7)
        self.use_llm_default = getattr(config, 'USE_LLM_DEFAULT', True)
    
    async def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate query input with external knowledge detection"""
        try:
            # Check if query is empty
            if not query or query.strip() == "":
                return False, "Query cannot be empty"
            
            # Check query length
            if len(query) > self.max_query_length:
                return False, f"Query too long. Maximum {self.max_query_length} characters"
            
            # Basic content validation
            query = query.strip()
            if len(query) < 3:
                return False, "Query must be at least 3 characters long"
            
            # Security: Check for potential injection attempts
            suspicious_patterns = ['<script', 'javascript:', 'eval(', 'exec(']
            query_lower = query.lower()
            for pattern in suspicious_patterns:
                if pattern in query_lower:
                    return False, "Query contains suspicious content"
            
            # Phase 2: External knowledge detection (configurable)
            external_knowledge_detected, external_reason = self.confidence_manager.detect_external_knowledge(query)
            if external_knowledge_detected:
                # Audit log: External knowledge blocked
                await self._audit_external_knowledge_blocked(query, external_reason)
                return False, f"Diese Frage erfordert externes Wissen außerhalb der verfügbaren Dokumente: {external_reason}"
            
            return True, "Valid query"
            
        except Exception as e:
            logger.error(f"Query validation error: {e}")
            return False, f"Validation failed: {str(e)}"
    
    async def search_documents(
        self,
        query: str,
        limit: Optional[int] = None,
        threshold: Optional[float] = None,
        use_llm: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Search documents with vector similarity"""
        try:
            # Validate query
            is_valid, validation_message = await self.validate_query(query)
            if not is_valid:
                logger.warning(f"Invalid query: {validation_message}")
                raise ValueError(validation_message)
            
            # Set defaults with intelligent multi-tier confidence system
            limit = limit or self.default_limit
            use_llm = use_llm if use_llm is not None else self.use_llm_default
            
            # Intelligent Multi-Tier Confidence System (configurable)
            confidence_tiers = self.confidence_manager.determine_confidence_tiers(query, use_llm)
            logger.info(f"Using confidence tiers: {confidence_tiers}")
            
            # Use very low threshold for vector search, then apply intelligent filtering
            vector_search_threshold = 0.05  # Very low to capture all potentially relevant results
            
            # Log query attempt
            logger.info(f"Query started: {query[:50]}... (use_llm={use_llm})")
            
            # Perform vector search with low threshold
            search_results = await self.vector_repo.search_similar_text(
                query=query,
                limit=limit,
                threshold=vector_search_threshold
            )
            
            # Intelligent Multi-Tier Processing
            processed_results = await self._process_results_with_confidence_tiers(
                search_results.items, confidence_tiers, query
            )
            
            if processed_results["should_refuse"]:
                # Audit log: Low confidence refused
                await self._audit_low_confidence_refused(
                    query, processed_results["max_similarity"], 
                    confidence_tiers["refusal_threshold"], len(search_results.items)
                )
                
                response_data = {
                    "query": query,
                    "results": [],
                    "total_found": 0,
                    "search_type": processed_results["response_type"],
                    "use_llm": False,
                    "processing_time": 0.1,
                    "refusal_reason": processed_results["message"],
                    "max_similarity": processed_results["max_similarity"],
                    "confidence_tier": processed_results["confidence_tier"]
                }
                
                logger.info(f"Query refused: {processed_results['reason']}")
                return response_data
            
            # Use processed results
            formatted_results = processed_results["results"]
            
            response_data = {
                "query": query,
                "results": formatted_results,
                "total_found": search_results.total_count,
                "search_type": processed_results["response_type"], 
                "use_llm": False,
                "processing_time": 0.1,
                "confidence_tier": processed_results["confidence_tier"],
                "max_similarity": processed_results["max_similarity"]
            }
            
            # Enhanced LLM processing based on confidence tier
            should_use_llm = (
                use_llm and 
                self.ollama_client and 
                formatted_results and
                processed_results["confidence_tier"] in ["high", "medium"]
            )
            
            if should_use_llm:
                try:
                    ai_response_data = await self._generate_ai_response(query, formatted_results)
                    response_data.update({
                        "ai_response": ai_response_data["answer"],
                        "ai_sources": ai_response_data["sources"],
                        "ai_confidence": ai_response_data["confidence"],
                        "source_count": ai_response_data["source_count"],
                        "search_type": "ai_enhanced",
                        "use_llm": True
                    })
                except Exception as e:
                    logger.warning(f"LLM processing failed, falling back to vector search: {e}")
            
            # Log successful query
            logger.info(f"Query completed: {len(formatted_results)} results ({response_data['search_type']})")
            
            return response_data
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            # Note: Audit logging would go here in production
            raise
    
    async def _generate_ai_response(self, query: str, search_results: List[Dict]) -> Dict[str, Any]:
        """Generate AI response using LLM with mandatory source citations and validation"""
        try:
            if not self.ollama_client:
                return {"answer": "LLM not available", "sources": [], "confidence": 0.0}
            
            # Zero Hallucination: Prepare context with explicit source tracking
            context_chunks = []
            source_references = []
            
            for i, result in enumerate(search_results[:5]):  # Use top 5 results
                doc_id = result['document_id']
                chunk_id = result.get('chunk_id', 'unknown')
                similarity = result['similarity']
                content = result['source_text']
                
                # Add numbered reference for citation
                ref_num = i + 1
                context_chunks.append(
                    f"[Quelle {ref_num}] Dokument-ID {doc_id}, Chunk {chunk_id} (Vertrauen: {similarity:.2f}):\n{content}"
                )
                
                source_references.append({
                    "reference_number": ref_num,
                    "document_id": doc_id,
                    "chunk_id": chunk_id,
                    "similarity": similarity,
                    "download_url": f"/api/v1/documents/{doc_id}/download",
                    "source_text": content  # Include for validation
                })
            
            context = "\n\n".join(context_chunks)
            
            # Zero Hallucination: Enhanced prompt with strict citation requirements
            enhanced_context = f"""
DOKUMENTE MIT QUELLENANGABEN:
{context}

WICHTIGE ANWEISUNGEN:
1. Beantworte NUR mit Informationen aus den obigen Dokumenten
2. Zitiere IMMER die Quelle als [Quelle X] im Text
3. Füge am Ende ALLE verwendeten Quellen mit Download-Links hinzu
4. Wenn keine passenden Informationen vorhanden: "Keine ausreichenden Informationen verfügbar"
"""
            
            # Generate response
            response = await self.ollama_client.generate_answer(
                query=query,
                context=enhanced_context
            )
            
            ai_answer = response.get('answer', 'No response generated')
            
            # Zero Hallucination: Add mandatory source footer
            source_footer = self._create_source_footer(source_references)
            complete_answer = f"{ai_answer}\n\n{source_footer}"
            
            # Phase 2: Response validation against source chunks
            validation_result = self._validate_ai_response(complete_answer, source_references)
            
            # If validation fails, return refusal instead of potentially hallucinated content
            if not validation_result.get("is_valid", False):
                logger.warning(f"AI response failed validation: {validation_result.get('reason', 'unknown')}")
                
                # Audit log: Response validation failed
                try:
                    await self._audit_response_validation_failed(
                        query, validation_result.get("reason", "unknown"), 
                        validation_result.get("confidence", 0.0)
                    )
                except Exception as audit_error:
                    logger.warning(f"Audit logging failed: {audit_error}")
                
                return {
                    "answer": "Die generierte Antwort konnte nicht ausreichend gegen die Quellen validiert werden. Bitte prüfen Sie die Originaldokumente direkt.",
                    "sources": source_references,
                    "confidence": 0.0,
                    "source_count": len(source_references),
                    "validation_failed": True,
                    "validation_reason": validation_result.get("reason", "unknown")
                }
            
            # Audit log: Successful LLM response generated
            try:
                await self._audit_llm_response_generated(
                    query, complete_answer, len(source_references),
                    validation_result.get("confidence", 0.0), True
                )
            except Exception as audit_error:
                logger.warning(f"Audit logging failed: {audit_error}")
            
            return {
                "answer": complete_answer,
                "sources": source_references,
                "confidence": min([r["similarity"] for r in source_references]) if source_references else 0.0,
                "source_count": len(source_references),
                "validation_passed": True,
                "validation_confidence": validation_result.get("confidence", 0.0)
            }
            
        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            raise
    
    async def smart_query(self, request: QueryRequest) -> SmartQueryResponse:
        """Enhanced query with smart routing and processing"""
        try:
            # Validate request
            is_valid, validation_message = await self.validate_query(request.query)
            if not is_valid:
                raise ValueError(validation_message)
            
            # Determine query strategy based on query type
            query_strategy = await self._analyze_query_intent(request.query)
            
            # Execute search based on strategy
            if query_strategy == "factual":
                # For factual queries, prioritize LLM
                results = await self.search_documents(
                    query=request.query,
                    limit=request.limit,
                    use_llm=True
                )
            elif query_strategy == "exploratory":
                # For exploratory queries, use vector search
                results = await self.search_documents(
                    query=request.query,
                    limit=request.limit,
                    use_llm=request.use_llm if hasattr(request, 'use_llm') else False
                )
            else:
                # Default strategy
                results = await self.search_documents(
                    query=request.query,
                    limit=request.limit,
                    use_llm=self.use_llm_default
                )
            
            # Format as SmartQueryResponse
            document_chunks = []
            for result in results["results"]:
                chunk = DocumentChunk(
                    document_id=result["document_id"],
                    text_content=result["content"],
                    metadata={**result["metadata"], "similarity_score": result["similarity"]}
                )
                document_chunks.append(chunk)
            
            response = SmartQueryResponse(
                query=request.query,
                answer=results.get("ai_response", ""),
                source_documents=document_chunks,
                total_documents_found=results["total_found"],
                search_strategy=query_strategy,
                processing_time=results["processing_time"],
                used_llm=results["use_llm"]
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Smart query failed: {e}")
            raise
    
    async def _analyze_query_intent(self, query: str) -> str:
        """Analyze query to determine best processing strategy"""
        try:
            query_lower = query.lower()
            
            # Factual query indicators
            factual_indicators = [
                'what is', 'who is', 'when did', 'where is', 'how much',
                'define', 'explain', 'describe', 'meaning of'
            ]
            
            # Exploratory query indicators
            exploratory_indicators = [
                'find documents about', 'search for', 'show me documents',
                'list documents', 'documents containing'
            ]
            
            for indicator in factual_indicators:
                if indicator in query_lower:
                    return "factual"
            
            for indicator in exploratory_indicators:
                if indicator in query_lower:
                    return "exploratory"
            
            # Default to factual if query is a question
            if any(word in query_lower for word in ['?', 'how', 'what', 'why', 'when', 'where', 'who']):
                return "factual"
            
            return "exploratory"
            
        except Exception as e:
            logger.error(f"Query intent analysis failed: {e}")
            return "default"
    
    async def get_document_chunks(
        self,
        document_id: int,
        page: int = 1,
        page_size: int = 20,
        search_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get chunks for a specific document"""
        try:
            # Verify document exists
            document = await self.doc_repo.get_by_id(document_id)
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # Get chunks from vector repository
            if search_query:
                # Search within document
                chunks_result = await self.vector_repo.search_in_document(
                    document_id=document_id,
                    query=search_query,
                    limit=page_size,
                    offset=(page - 1) * page_size
                )
            else:
                # Get all chunks for document
                chunks_result = await self.vector_repo.get_document_chunks(
                    document_id=document_id,
                    limit=page_size,
                    offset=(page - 1) * page_size
                )
            
            # Format response
            formatted_chunks = []
            for chunk in chunks_result.items:
                formatted_chunks.append({
                    "chunk_id": getattr(chunk, 'chunk_id', None),
                    "content": chunk.content,
                    "start_index": getattr(chunk, 'start_index', 0),
                    "end_index": getattr(chunk, 'end_index', len(chunk.content)),
                    "metadata": chunk.metadata or {}
                })
            
            return {
                "document_id": document_id,
                "chunks": formatted_chunks,
                "page": page,
                "page_size": page_size,
                "total_chunks": chunks_result.total_count,
                "search_query": search_query
            }
            
        except Exception as e:
            logger.error(f"Error getting document chunks: {e}")
            raise
    
    async def get_query_suggestions(self, partial_query: str) -> List[str]:
        """Get query suggestions based on partial input"""
        try:
            # This is a placeholder for query suggestion logic
            # Could be enhanced with:
            # - Analysis of previous successful queries
            # - Document content analysis
            # - Common query patterns
            
            suggestions = []
            
            if len(partial_query) >= 3:
                # Generate basic suggestions
                base_suggestions = [
                    f"What is {partial_query}?",
                    f"Find documents about {partial_query}",
                    f"Explain {partial_query}",
                    f"Show me information on {partial_query}"
                ]
                suggestions.extend(base_suggestions)
            
            return suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            logger.error(f"Error generating query suggestions: {e}")
            return []
    
    # Confidence tier determination now handled by ConfidenceManager
    
    async def _process_results_with_confidence_tiers(
        self, 
        search_items: List[Any], 
        confidence_tiers: Dict[str, float], 
        query: str
    ) -> Dict[str, Any]:
        """Process search results using intelligent multi-tier confidence system"""
        try:
            if not search_items:
                return {
                    "should_refuse": True,
                    "response_type": "no_results",
                    "confidence_tier": "none",
                    "max_similarity": 0.0,
                    "message": "Dazu finde ich keine Informationen in den verfügbaren Dokumenten.",
                    "reason": "No search results found",
                    "results": []
                }
            
            # Analyze all results and find best match
            all_results = []
            max_similarity = 0.0
            
            for result in search_items:
                content = result.text_content or ""
                similarity = result.metadata.get("similarity_score", 0.0)
                max_similarity = max(max_similarity, similarity)
                
                all_results.append({
                    "document_id": result.document_id,
                    "content": content[:500] + "..." if len(content) > 500 else content,
                    "similarity": similarity,
                    "metadata": result.metadata or {},
                    "chunk_id": result.id,
                    "source_text": content  # Full text for citation
                })
            
            # Determine confidence tier based on best match
            if max_similarity >= confidence_tiers["high_threshold"]:
                confidence_tier = "high"
                response_type = "high_confidence"
                filtered_results = [r for r in all_results if r["similarity"] >= confidence_tiers["medium_threshold"]]
                
            elif max_similarity >= confidence_tiers["medium_threshold"]:
                confidence_tier = "medium" 
                response_type = "medium_confidence"
                filtered_results = [r for r in all_results if r["similarity"] >= confidence_tiers["low_threshold"]]
                
            elif max_similarity >= confidence_tiers["low_threshold"]:
                confidence_tier = "low"
                response_type = "low_confidence_suggestions"
                filtered_results = [r for r in all_results if r["similarity"] >= confidence_tiers["refusal_threshold"]]
                
            else:
                # Below refusal threshold
                return {
                    "should_refuse": True,
                    "response_type": "insufficient_confidence",
                    "confidence_tier": "rejected",
                    "max_similarity": max_similarity,
                    "message": f"Die gefundenen Informationen haben nicht die erforderliche Zuverlässigkeit (Vertrauen: {max_similarity:.2f}). Für präzise Antworten benötige ich eindeutigere Dokumentinhalte zu Ihrer Frage.",
                    "reason": f"Max similarity {max_similarity:.3f} below refusal threshold {confidence_tiers['refusal_threshold']:.3f}",
                    "results": []
                }
            
            logger.info(f"Confidence tier: {confidence_tier}, max similarity: {max_similarity:.3f}, returning {len(filtered_results)} results")
            
            return {
                "should_refuse": False,
                "response_type": response_type,
                "confidence_tier": confidence_tier,
                "max_similarity": max_similarity,
                "results": filtered_results[:10],  # Limit to top 10
                "total_processed": len(all_results)
            }
            
        except Exception as e:
            logger.error(f"Error processing results with confidence tiers: {e}")
            return {
                "should_refuse": True,
                "response_type": "processing_error",
                "confidence_tier": "error",
                "max_similarity": 0.0,
                "message": "Ein Fehler ist bei der Verarbeitung aufgetreten.",
                "reason": f"Processing error: {str(e)}",
                "results": []
            }
    
    def _generate_refusal_message(self, query: str, max_similarity: float, total_results: int) -> str:
        """Generate appropriate refusal message based on search results"""
        if total_results == 0:
            return "Dazu finde ich keine Informationen in den verfügbaren Dokumenten."
        elif max_similarity < 0.2:
            return "Dazu finde ich keine ausreichend relevanten Informationen in den verfügbaren Dokumenten."
        else:
            return f"Die gefundenen Informationen haben nicht die erforderliche Zuverlässigkeit (Vertrauen: {max_similarity:.2f}). Für präzise Antworten benötige ich eindeutigere Dokumentinhalte zu Ihrer Frage."
    
    def _create_source_footer(self, source_references: List[Dict]) -> str:
        """Create formatted source citations footer"""
        if not source_references:
            return ""
        
        footer_lines = ["📚 VERWENDETE QUELLEN:"]
        
        for ref in source_references:
            footer_lines.append(
                f"[Quelle {ref['reference_number']}] Dokument-ID {ref['document_id']}, "
                f"Chunk {ref['chunk_id']} (Vertrauen: {ref['similarity']:.2f}) - "
                f"Download: {ref['download_url']}"
            )
        
        return "\n".join(footer_lines)
    
    def _validate_ai_response(self, ai_response: str, source_chunks: List[Dict]) -> Dict[str, Any]:
        """Validate AI response against source chunks using validation service"""
        try:
            # Import here to avoid circular imports
            from .validation_service import ValidationService
            
            # Create validation service instance
            validator = ValidationService()
            
            # Perform response content validation
            validation_result = validator.validate_response_content(
                llm_response=ai_response,
                source_chunks=source_chunks
            )
            
            # Also validate source citations
            citation_validation = validator.validate_source_citations(
                llm_response=ai_response,
                source_chunks=source_chunks
            )
            
            # Combine validation results
            overall_valid = (
                validation_result.get("is_valid", False) and 
                citation_validation.get("is_valid", False)
            )
            
            return {
                "is_valid": overall_valid,
                "confidence": validation_result.get("confidence", 0.0),
                "reason": validation_result.get("reason", "unknown"),
                "content_validation": validation_result,
                "citation_validation": citation_validation
            }
            
        except Exception as e:
            logger.error(f"Response validation failed: {e}")
            return {
                "is_valid": False,
                "confidence": 0.0,
                "reason": f"validation_error: {str(e)}"
            }
    
    # External knowledge detection now handled by ConfidenceManager
    
    # Audit logging helper methods
    async def _audit_external_knowledge_blocked(self, query: str, external_reason: str):
        """Log external knowledge blocking event"""
        try:
            from ..repositories.audit_repository import log_external_knowledge_blocked
            await log_external_knowledge_blocked(
                audit_repo=self.audit_repo,
                user_id="anonymous",  # Would be real user ID in production
                query_text=query,
                external_reason=external_reason,
                user_ip="127.0.0.1"  # Would be real IP in production
            )
        except Exception as e:
            logger.warning(f"Failed to log external knowledge blocked: {e}")
    
    async def _audit_low_confidence_refused(self, query: str, max_similarity: float, threshold: float, result_count: int):
        """Log low confidence refusal event"""
        try:
            from ..repositories.audit_repository import log_low_confidence_refused
            await log_low_confidence_refused(
                audit_repo=self.audit_repo,
                user_id="anonymous",
                query_text=query,
                max_similarity=max_similarity,
                threshold=threshold,
                result_count=result_count,
                user_ip="127.0.0.1"
            )
        except Exception as e:
            logger.warning(f"Failed to log low confidence refused: {e}")
    
    async def _audit_response_validation_failed(self, query: str, validation_reason: str, confidence_score: float):
        """Log response validation failure event"""
        try:
            from ..repositories.audit_repository import log_response_validation_failed
            await log_response_validation_failed(
                audit_repo=self.audit_repo,
                user_id="anonymous",
                query_text=query,
                validation_reason=validation_reason,
                confidence_score=confidence_score,
                user_ip="127.0.0.1"
            )
        except Exception as e:
            logger.warning(f"Failed to log response validation failed: {e}")
    
    async def _audit_llm_response_generated(self, query: str, response: str, source_count: int, confidence: float, validation_passed: bool):
        """Log successful LLM response generation"""
        try:
            from ..repositories.audit_repository import log_llm_response_generated
            await log_llm_response_generated(
                audit_repo=self.audit_repo,
                user_id="anonymous",
                query_text=query,
                response_text=response,
                source_count=source_count,
                confidence_score=confidence,
                validation_passed=validation_passed,
                user_ip="127.0.0.1",
                processing_time_ms=100.0  # Would be actual timing in production
            )
        except Exception as e:
            logger.warning(f"Failed to log LLM response generated: {e}")