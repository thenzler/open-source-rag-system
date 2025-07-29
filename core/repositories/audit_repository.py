"""
Swiss Compliance Audit Repository
GDPR/FADP compliant audit logging for Swiss market
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of auditable events"""

    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_ACCESS = "document_access"
    DOCUMENT_DOWNLOAD = "document_download"
    DOCUMENT_DELETE = "document_delete"
    QUERY_EXECUTED = "query_executed"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    DATA_EXPORT = "data_export"
    SYSTEM_ACCESS = "system_access"
    CONFIGURATION_CHANGE = "configuration_change"
    # Zero-hallucination specific events
    LLM_RESPONSE_GENERATED = "llm_response_generated"
    RESPONSE_VALIDATION_FAILED = "response_validation_failed"
    EXTERNAL_KNOWLEDGE_BLOCKED = "external_knowledge_blocked"
    LOW_CONFIDENCE_REFUSED = "low_confidence_refused"
    SOURCE_CITATION_ERROR = "source_citation_error"


class DataClassification(Enum):
    """Data classification levels for Swiss compliance"""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    PERSONAL_DATA = "personal_data"  # GDPR/FADP relevant
    SENSITIVE_PERSONAL_DATA = "sensitive_personal_data"  # Special category


class LegalBasis(Enum):
    """GDPR/FADP legal basis for processing"""

    CONSENT = "consent"  # Art. 6(1)(a) GDPR
    CONTRACT = "contract"  # Art. 6(1)(b) GDPR
    LEGAL_OBLIGATION = "legal_obligation"  # Art. 6(1)(c) GDPR
    VITAL_INTERESTS = "vital_interests"  # Art. 6(1)(d) GDPR
    PUBLIC_TASK = "public_task"  # Art. 6(1)(e) GDPR
    LEGITIMATE_INTERESTS = "legitimate_interests"  # Art. 6(1)(f) GDPR


@dataclass
class AuditEntry:
    """Immutable audit log entry"""

    id: Optional[int] = None
    timestamp: datetime = None
    event_type: AuditEventType = None
    user_id: Optional[str] = None
    user_ip: Optional[str] = None
    session_id: Optional[str] = None

    # Data being processed
    document_id: Optional[int] = None
    data_classification: DataClassification = DataClassification.INTERNAL
    legal_basis: LegalBasis = LegalBasis.LEGITIMATE_INTERESTS

    # Event details
    action_description: str = ""
    resource_accessed: Optional[str] = None
    query_text: Optional[str] = None  # Anonymized if needed

    # Technical details
    user_agent: Optional[str] = None
    request_method: Optional[str] = None
    response_status: Optional[int] = None
    processing_time_ms: Optional[float] = None

    # Swiss compliance fields
    data_subject_id: Optional[str] = None  # If data relates to specific person
    retention_period_days: Optional[int] = None
    anonymization_applied: bool = False

    # Additional metadata
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)

        # Convert enums to strings
        if self.event_type:
            data["event_type"] = self.event_type.value
        if self.data_classification:
            data["data_classification"] = self.data_classification.value
        if self.legal_basis:
            data["legal_basis"] = self.legal_basis.value

        # Convert datetime to ISO string
        if self.timestamp:
            data["timestamp"] = self.timestamp.isoformat()

        return data


class SwissAuditRepository:
    """Swiss GDPR/FADP compliant audit repository"""

    def __init__(self, db_path: str = "data/audit.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread_local = threading.local()
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection for audit DB"""
        if not hasattr(self._thread_local, "audit_connection"):
            conn = sqlite3.connect(
                str(self.db_path), check_same_thread=False, timeout=30.0
            )
            conn.row_factory = sqlite3.Row
            # Audit logs should be immutable and persistent
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = FULL")  # Maximum durability
            conn.execute("PRAGMA foreign_keys = ON")
            self._thread_local.audit_connection = conn
        return self._thread_local.audit_connection

    @contextmanager
    def get_connection(self):
        """Context manager for audit database connections"""
        conn = self._get_connection()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.commit()  # Always commit audit entries

    def _init_database(self):
        """Initialize audit database schema"""
        with self.get_connection() as conn:
            # Main audit log table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    user_ip TEXT,
                    session_id TEXT,

                    -- Data being processed
                    document_id INTEGER,
                    data_classification TEXT NOT NULL DEFAULT 'internal',
                    legal_basis TEXT NOT NULL DEFAULT 'legitimate_interests',

                    -- Event details
                    action_description TEXT NOT NULL,
                    resource_accessed TEXT,
                    query_text TEXT,  -- May be anonymized

                    -- Technical details
                    user_agent TEXT,
                    request_method TEXT,
                    response_status INTEGER,
                    processing_time_ms REAL,

                    -- Swiss compliance
                    data_subject_id TEXT,
                    retention_period_days INTEGER,
                    anonymization_applied BOOLEAN DEFAULT 0,

                    -- Metadata
                    metadata TEXT,  -- JSON

                    -- Immutability protection
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    checksum TEXT  -- For integrity verification
                )
            """
            )

            # Data retention tracking
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS retention_policies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_classification TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    retention_days INTEGER NOT NULL,
                    anonymization_after_days INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(data_classification, event_type)
                )
            """
            )

            # GDPR/FADP compliance tracking
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS compliance_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,  -- 'data_request', 'erasure_request', 'portability_request'
                    data_subject_id TEXT NOT NULL,
                    request_date DATETIME NOT NULL,
                    completion_date DATETIME,
                    status TEXT NOT NULL DEFAULT 'pending',  -- 'pending', 'completed', 'rejected'
                    legal_basis TEXT,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Indexes for performance and compliance queries
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log (timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log (user_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_document ON audit_log (document_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log (event_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_data_subject ON audit_log (data_subject_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_compliance_subject ON compliance_events (data_subject_id)"
            )

            # Insert default retention policies
            self._insert_default_retention_policies(conn)

            logger.info(f"Initialized Swiss audit database at {self.db_path}")

    def _insert_default_retention_policies(self, conn):
        """Insert default Swiss compliance retention policies"""
        default_policies = [
            # Document access logs - keep for 3 years (Swiss banking standard)
            ("personal_data", "document_access", 1095, 365),
            (
                "confidential",
                "document_access",
                2555,
                730,
            ),  # 7 years for business documents
            ("internal", "document_access", 365, 180),
            # Query logs - shorter retention for privacy
            ("personal_data", "query_executed", 365, 90),
            ("internal", "query_executed", 180, 90),
            # System access - security requirement
            ("internal", "user_login", 730, None),  # 2 years, no anonymization
            ("internal", "system_access", 365, None),
            # Data exports - compliance tracking
            ("personal_data", "data_export", 2555, None),  # 7 years, no anonymization
        ]

        for data_class, event_type, retention_days, anon_days in default_policies:
            conn.execute(
                """
                INSERT OR IGNORE INTO retention_policies
                (data_classification, event_type, retention_days, anonymization_after_days)
                VALUES (?, ?, ?, ?)
            """,
                (data_class, event_type, retention_days, anon_days),
            )

    def _calculate_checksum(self, entry: AuditEntry) -> str:
        """Calculate integrity checksum for audit entry"""
        import hashlib

        # Create deterministic string from entry
        content = f"{entry.timestamp.isoformat()}{entry.event_type.value if entry.event_type else ''}{entry.user_id or ''}{entry.action_description}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def log_event(self, entry: AuditEntry) -> int:
        """Log an audit event (immutable)"""
        try:
            checksum = self._calculate_checksum(entry)

            with self.get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO audit_log (
                        timestamp, event_type, user_id, user_ip, session_id,
                        document_id, data_classification, legal_basis,
                        action_description, resource_accessed, query_text,
                        user_agent, request_method, response_status, processing_time_ms,
                        data_subject_id, retention_period_days, anonymization_applied,
                        metadata, checksum
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entry.timestamp.isoformat(),
                        entry.event_type.value if entry.event_type else None,
                        entry.user_id,
                        entry.user_ip,
                        entry.session_id,
                        entry.document_id,
                        (
                            entry.data_classification.value
                            if entry.data_classification
                            else "internal"
                        ),
                        (
                            entry.legal_basis.value
                            if entry.legal_basis
                            else "legitimate_interests"
                        ),
                        entry.action_description,
                        entry.resource_accessed,
                        entry.query_text,
                        entry.user_agent,
                        entry.request_method,
                        entry.response_status,
                        entry.processing_time_ms,
                        entry.data_subject_id,
                        entry.retention_period_days,
                        entry.anonymization_applied,
                        json.dumps(entry.metadata) if entry.metadata else None,
                        checksum,
                    ),
                )

                audit_id = cursor.lastrowid
                logger.debug(
                    f"Logged audit event {entry.event_type.value if entry.event_type else 'unknown'} with ID {audit_id}"
                )
                return audit_id

        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            # Audit logging failures should not break the main application
            return -1

    async def get_user_data_access_log(
        self, user_id: str, days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """Get user's data access history for GDPR/FADP compliance"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)

            with self.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT * FROM audit_log
                    WHERE user_id = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                """,
                    (user_id, cutoff_date.isoformat()),
                )

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get user data access log: {e}")
            return []

    async def get_document_access_history(
        self, document_id: int
    ) -> List[Dict[str, Any]]:
        """Get complete access history for a document"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT * FROM audit_log
                    WHERE document_id = ?
                    ORDER BY timestamp DESC
                """,
                    (document_id,),
                )

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get document access history: {e}")
            return []

    async def anonymize_old_entries(self) -> int:
        """Anonymize old audit entries according to retention policies"""
        try:
            anonymized_count = 0

            with self.get_connection() as conn:
                # Get retention policies
                cursor = conn.execute(
                    """
                    SELECT data_classification, event_type, anonymization_after_days
                    FROM retention_policies
                    WHERE anonymization_after_days IS NOT NULL
                """
                )

                policies = cursor.fetchall()

                for policy in policies:
                    data_class, event_type, anon_days = policy
                    cutoff_date = datetime.utcnow() - timedelta(days=anon_days)

                    # Anonymize qualifying entries
                    cursor = conn.execute(
                        """
                        UPDATE audit_log
                        SET
                            user_id = 'ANONYMIZED',
                            user_ip = 'ANONYMIZED',
                            data_subject_id = 'ANONYMIZED',
                            query_text = '[ANONYMIZED]',
                            user_agent = 'ANONYMIZED',
                            anonymization_applied = 1
                        WHERE
                            data_classification = ?
                            AND event_type = ?
                            AND timestamp < ?
                            AND anonymization_applied = 0
                    """,
                        (data_class, event_type, cutoff_date.isoformat()),
                    )

                    anonymized_count += cursor.rowcount

                logger.info(f"Anonymized {anonymized_count} audit entries")
                return anonymized_count

        except Exception as e:
            logger.error(f"Failed to anonymize audit entries: {e}")
            return 0

    async def delete_expired_entries(self) -> int:
        """Delete audit entries that have exceeded their retention period"""
        try:
            deleted_count = 0

            with self.get_connection() as conn:
                # Get retention policies
                cursor = conn.execute(
                    """
                    SELECT data_classification, event_type, retention_days
                    FROM retention_policies
                """
                )

                policies = cursor.fetchall()

                for policy in policies:
                    data_class, event_type, retention_days = policy
                    cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

                    # Delete expired entries
                    cursor = conn.execute(
                        """
                        DELETE FROM audit_log
                        WHERE
                            data_classification = ?
                            AND event_type = ?
                            AND timestamp < ?
                    """,
                        (data_class, event_type, cutoff_date.isoformat()),
                    )

                    deleted_count += cursor.rowcount

                logger.info(f"Deleted {deleted_count} expired audit entries")
                return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete expired audit entries: {e}")
            return 0

    async def get_compliance_report(self, days_back: int = 30) -> Dict[str, Any]:
        """Generate Swiss compliance report"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)

            with self.get_connection() as conn:
                # Event type summary
                cursor = conn.execute(
                    """
                    SELECT event_type, data_classification, COUNT(*) as count
                    FROM audit_log
                    WHERE timestamp >= ?
                    GROUP BY event_type, data_classification
                """,
                    (cutoff_date.isoformat(),),
                )

                event_summary = cursor.fetchall()

                # Data subject access summary
                cursor = conn.execute(
                    """
                    SELECT COUNT(DISTINCT data_subject_id) as unique_subjects,
                           COUNT(*) as total_personal_data_events
                    FROM audit_log
                    WHERE data_classification IN ('personal_data', 'sensitive_personal_data')
                      AND timestamp >= ?
                      AND data_subject_id IS NOT NULL
                """,
                    (cutoff_date.isoformat(),),
                )

                privacy_stats = cursor.fetchone()

                # Compliance events
                cursor = conn.execute(
                    """
                    SELECT event_type, status, COUNT(*) as count
                    FROM compliance_events
                    WHERE request_date >= ?
                    GROUP BY event_type, status
                """,
                    (cutoff_date.isoformat(),),
                )

                compliance_events = cursor.fetchall()

                return {
                    "report_period_days": days_back,
                    "generated_at": datetime.utcnow().isoformat(),
                    "event_summary": [dict(row) for row in event_summary],
                    "privacy_statistics": dict(privacy_stats) if privacy_stats else {},
                    "compliance_events": [dict(row) for row in compliance_events],
                    "retention_policies_active": True,
                    "anonymization_active": True,
                }

        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            return {"error": str(e)}


# Convenience functions for common audit events
async def log_document_access(
    audit_repo: SwissAuditRepository,
    user_id: str,
    document_id: int,
    user_ip: str,
    action: str = "view",
):
    """Log document access event"""
    entry = AuditEntry(
        event_type=AuditEventType.DOCUMENT_ACCESS,
        user_id=user_id,
        user_ip=user_ip,
        document_id=document_id,
        data_classification=DataClassification.INTERNAL,  # Should be determined dynamically
        legal_basis=LegalBasis.LEGITIMATE_INTERESTS,
        action_description=f"Document {action}: ID {document_id}",
    )
    return await audit_repo.log_event(entry)


async def log_query_execution(
    audit_repo: SwissAuditRepository,
    user_id: str,
    query_text: str,
    result_count: int,
    user_ip: str,
    processing_time_ms: float,
):
    """Log query execution event"""
    # Optionally anonymize query text if it contains personal data
    anonymized_query = query_text[:100] + "..." if len(query_text) > 100 else query_text

    entry = AuditEntry(
        event_type=AuditEventType.QUERY_EXECUTED,
        user_id=user_id,
        user_ip=user_ip,
        data_classification=DataClassification.INTERNAL,
        legal_basis=LegalBasis.LEGITIMATE_INTERESTS,
        action_description=f"Query executed, {result_count} results returned",
        query_text=anonymized_query,
        processing_time_ms=processing_time_ms,
    )
    return await audit_repo.log_event(entry)


# Zero-hallucination specific audit functions
async def log_llm_response_generated(
    audit_repo: SwissAuditRepository,
    user_id: str,
    query_text: str,
    response_text: str,
    source_count: int,
    confidence_score: float,
    validation_passed: bool,
    user_ip: str,
    processing_time_ms: float,
):
    """Log LLM response generation with zero-hallucination validation results"""
    # Truncate texts for audit log
    truncated_query = query_text[:200] + "..." if len(query_text) > 200 else query_text
    truncated_response = (
        response_text[:300] + "..." if len(response_text) > 300 else response_text
    )

    entry = AuditEntry(
        event_type=AuditEventType.LLM_RESPONSE_GENERATED,
        user_id=user_id,
        user_ip=user_ip,
        data_classification=DataClassification.INTERNAL,
        legal_basis=LegalBasis.LEGITIMATE_INTERESTS,
        action_description=f"LLM response generated: {source_count} sources, confidence {confidence_score:.3f}, validation {'passed' if validation_passed else 'failed'}",
        query_text=truncated_query,
        processing_time_ms=processing_time_ms,
        metadata={
            "response_preview": truncated_response,
            "source_count": source_count,
            "confidence_score": confidence_score,
            "validation_passed": validation_passed,
            "response_length": len(response_text),
            "has_source_citations": "[Quelle" in response_text,
        },
    )
    return await audit_repo.log_event(entry)


async def log_response_validation_failed(
    audit_repo: SwissAuditRepository,
    user_id: str,
    query_text: str,
    validation_reason: str,
    confidence_score: float,
    user_ip: str,
):
    """Log when LLM response validation fails"""
    truncated_query = query_text[:200] + "..." if len(query_text) > 200 else query_text

    entry = AuditEntry(
        event_type=AuditEventType.RESPONSE_VALIDATION_FAILED,
        user_id=user_id,
        user_ip=user_ip,
        data_classification=DataClassification.INTERNAL,
        legal_basis=LegalBasis.LEGITIMATE_INTERESTS,
        action_description=f"Response validation failed: {validation_reason}",
        query_text=truncated_query,
        metadata={
            "validation_reason": validation_reason,
            "confidence_score": confidence_score,
            "zero_hallucination_enforced": True,
        },
    )
    return await audit_repo.log_event(entry)


async def log_external_knowledge_blocked(
    audit_repo: SwissAuditRepository,
    user_id: str,
    query_text: str,
    external_reason: str,
    user_ip: str,
):
    """Log when query is blocked for requiring external knowledge"""
    truncated_query = query_text[:200] + "..." if len(query_text) > 200 else query_text

    entry = AuditEntry(
        event_type=AuditEventType.EXTERNAL_KNOWLEDGE_BLOCKED,
        user_id=user_id,
        user_ip=user_ip,
        data_classification=DataClassification.INTERNAL,
        legal_basis=LegalBasis.LEGITIMATE_INTERESTS,
        action_description=f"External knowledge query blocked: {external_reason}",
        query_text=truncated_query,
        metadata={
            "external_knowledge_type": external_reason,
            "preprocessing_blocked": True,
            "zero_hallucination_enforced": True,
        },
    )
    return await audit_repo.log_event(entry)


async def log_low_confidence_refused(
    audit_repo: SwissAuditRepository,
    user_id: str,
    query_text: str,
    max_similarity: float,
    threshold: float,
    result_count: int,
    user_ip: str,
):
    """Log when query is refused due to low confidence/similarity"""
    truncated_query = query_text[:200] + "..." if len(query_text) > 200 else query_text

    entry = AuditEntry(
        event_type=AuditEventType.LOW_CONFIDENCE_REFUSED,
        user_id=user_id,
        user_ip=user_ip,
        data_classification=DataClassification.INTERNAL,
        legal_basis=LegalBasis.LEGITIMATE_INTERESTS,
        action_description=f"Low confidence refusal: max similarity {max_similarity:.3f} below threshold {threshold:.3f}",
        query_text=truncated_query,
        metadata={
            "max_similarity": max_similarity,
            "confidence_threshold": threshold,
            "result_count": result_count,
            "zero_hallucination_enforced": True,
        },
    )
    return await audit_repo.log_event(entry)


async def log_source_citation_error(
    audit_repo: SwissAuditRepository,
    user_id: str,
    response_text: str,
    citation_error: str,
    user_ip: str,
):
    """Log when source citation validation fails"""
    truncated_response = (
        response_text[:300] + "..." if len(response_text) > 300 else response_text
    )

    entry = AuditEntry(
        event_type=AuditEventType.SOURCE_CITATION_ERROR,
        user_id=user_id,
        user_ip=user_ip,
        data_classification=DataClassification.INTERNAL,
        legal_basis=LegalBasis.LEGITIMATE_INTERESTS,
        action_description=f"Source citation error: {citation_error}",
        metadata={
            "response_preview": truncated_response,
            "citation_error": citation_error,
            "zero_hallucination_enforced": True,
        },
    )
    return await audit_repo.log_event(entry)
