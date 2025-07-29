"""
GDPR/DSG Compliance Service
Handles Swiss Data Protection and GDPR compliance requirements
"""
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class DataCategory(Enum):
    """Categories of personal data"""
    PERSONAL_IDENTIFIERS = "personal_identifiers"  # Names, email, phone
    BIOMETRIC_DATA = "biometric_data"  # Fingerprints, photos
    FINANCIAL_DATA = "financial_data"  # Bank details, credit cards
    HEALTH_DATA = "health_data"  # Medical records
    BEHAVIORAL_DATA = "behavioral_data"  # Preferences, usage patterns
    TECHNICAL_DATA = "technical_data"  # IP addresses, device info
    COMMUNICATION_DATA = "communication_data"  # Messages, documents

class ProcessingPurpose(Enum):
    """Legal purposes for data processing"""
    CONTRACT = "contract"  # Contractual necessity
    LEGAL_OBLIGATION = "legal_obligation"  # Legal compliance
    LEGITIMATE_INTEREST = "legitimate_interest"  # Legitimate business interest
    CONSENT = "consent"  # Explicit user consent
    PUBLIC_INTEREST = "public_interest"  # Public task
    VITAL_INTEREST = "vital_interest"  # Protection of vital interests

class DataSubjectRight(Enum):
    """Data subject rights under GDPR/DSG"""
    ACCESS = "access"  # Right to access
    RECTIFICATION = "rectification"  # Right to rectification
    ERASURE = "erasure"  # Right to erasure (right to be forgotten)
    PORTABILITY = "portability"  # Right to data portability
    RESTRICT_PROCESSING = "restrict_processing"  # Right to restrict processing
    OBJECT = "object"  # Right to object
    WITHDRAW_CONSENT = "withdraw_consent"  # Right to withdraw consent

@dataclass
class DataProcessingRecord:
    """Record of data processing activity"""
    id: str
    tenant_id: str
    data_category: DataCategory
    processing_purpose: ProcessingPurpose
    data_subject_id: Optional[str]  # Hashed identifier
    processed_at: datetime
    retention_until: datetime
    legal_basis: str
    data_source: str  # Document, query, upload, etc.
    anonymized: bool = False
    exported: bool = False
    deleted: bool = False
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ConsentRecord:
    """Record of data subject consent"""
    id: str
    tenant_id: str
    data_subject_id: str  # Hashed identifier
    consent_type: str
    given_at: datetime
    withdrawn_at: Optional[datetime]
    purpose: ProcessingPurpose
    data_categories: List[DataCategory]
    valid: bool = True
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class DataSubjectRequest:
    """Data subject rights request"""
    id: str
    tenant_id: str
    data_subject_id: str
    request_type: DataSubjectRight
    requested_at: datetime
    processed_at: Optional[datetime]
    completed_at: Optional[datetime]
    status: str  # pending, processing, completed, rejected
    request_data: Dict[str, Any]
    response_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class SwissDataProtectionService:
    """Swiss Data Protection (DSG) and GDPR compliance service"""
    
    def __init__(
        self,
        storage_path: str = "data/compliance",
        enable_audit_logging: bool = True,
        data_residency_region: str = "CH"
    ):
        self.storage_path = Path(storage_path)
        self.enable_audit_logging = enable_audit_logging
        self.data_residency_region = data_residency_region
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Storage for compliance records
        self.processing_records: Dict[str, DataProcessingRecord] = {}
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.data_subject_requests: Dict[str, DataSubjectRequest] = {}
        
        # Swiss data protection specific settings
        self.swiss_retention_defaults = {
            DataCategory.PERSONAL_IDENTIFIERS: timedelta(days=2555),  # 7 years
            DataCategory.FINANCIAL_DATA: timedelta(days=3650),  # 10 years
            DataCategory.HEALTH_DATA: timedelta(days=2555),  # 7 years
            DataCategory.BEHAVIORAL_DATA: timedelta(days=365),  # 1 year
            DataCategory.TECHNICAL_DATA: timedelta(days=90),  # 3 months
            DataCategory.COMMUNICATION_DATA: timedelta(days=2555),  # 7 years
        }
        
        self._load_compliance_data()
        logger.info(f"Swiss Data Protection Service initialized (region: {data_residency_region})")
    
    def hash_data_subject_id(self, identifier: str) -> str:
        """Create a hashed identifier for data subjects (pseudonymization)"""
        # Use SHA-256 with salt for pseudonymization
        salt = "swiss_rag_system_salt_2025"  # Should be configurable
        return hashlib.sha256(f"{salt}:{identifier}".encode()).hexdigest()
    
    async def record_data_processing(
        self,
        tenant_id: str,
        data_category: DataCategory,
        processing_purpose: ProcessingPurpose,
        data_subject_identifier: Optional[str] = None,
        data_source: str = "unknown",
        legal_basis: str = "legitimate_interest",
        custom_retention_days: Optional[int] = None
    ) -> str:
        """Record a data processing activity"""
        
        # Hash the data subject identifier for privacy
        data_subject_id = None
        if data_subject_identifier:
            data_subject_id = self.hash_data_subject_id(data_subject_identifier)
        
        # Calculate retention period
        if custom_retention_days:
            retention_until = datetime.now() + timedelta(days=custom_retention_days)
        else:
            default_retention = self.swiss_retention_defaults.get(
                data_category, 
                timedelta(days=365)
            )
            retention_until = datetime.now() + default_retention
        
        # Create processing record
        record_id = f"proc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tenant_id}"
        record = DataProcessingRecord(
            id=record_id,
            tenant_id=tenant_id,
            data_category=data_category,
            processing_purpose=processing_purpose,
            data_subject_id=data_subject_id,
            processed_at=datetime.now(),
            retention_until=retention_until,
            legal_basis=legal_basis,
            data_source=data_source
        )
        
        # Store record
        self.processing_records[record_id] = record
        
        # Audit log
        if self.enable_audit_logging:
            await self._audit_log("data_processing_recorded", {
                "record_id": record_id,
                "tenant_id": tenant_id,
                "data_category": data_category.value,
                "legal_basis": legal_basis
            })
        
        # Persist to storage
        await self._persist_compliance_data()
        
        logger.info(f"Data processing recorded: {record_id}")
        return record_id
    
    async def record_consent(
        self,
        tenant_id: str,
        data_subject_identifier: str,
        consent_type: str,
        purpose: ProcessingPurpose,
        data_categories: List[DataCategory]
    ) -> str:
        """Record data subject consent"""
        
        data_subject_id = self.hash_data_subject_id(data_subject_identifier)
        
        consent_id = f"consent_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tenant_id}"
        consent = ConsentRecord(
            id=consent_id,
            tenant_id=tenant_id,
            data_subject_id=data_subject_id,
            consent_type=consent_type,
            given_at=datetime.now(),
            purpose=purpose,
            data_categories=data_categories
        )
        
        self.consent_records[consent_id] = consent
        
        if self.enable_audit_logging:
            await self._audit_log("consent_recorded", {
                "consent_id": consent_id,
                "tenant_id": tenant_id,
                "consent_type": consent_type,
                "data_categories": [cat.value for cat in data_categories]
            })
        
        await self._persist_compliance_data()
        logger.info(f"Consent recorded: {consent_id}")
        return consent_id
    
    async def withdraw_consent(self, consent_id: str) -> bool:
        """Withdraw previously given consent"""
        consent = self.consent_records.get(consent_id)
        if not consent or not consent.valid:
            return False
        
        consent.withdrawn_at = datetime.now()
        consent.valid = False
        
        if self.enable_audit_logging:
            await self._audit_log("consent_withdrawn", {
                "consent_id": consent_id,
                "tenant_id": consent.tenant_id,
                "withdrawn_at": consent.withdrawn_at.isoformat()
            })
        
        await self._persist_compliance_data()
        logger.info(f"Consent withdrawn: {consent_id}")
        return True
    
    async def submit_data_subject_request(
        self,
        tenant_id: str,
        data_subject_identifier: str,
        request_type: DataSubjectRight,
        request_data: Dict[str, Any]
    ) -> str:
        """Submit a data subject rights request"""
        
        data_subject_id = self.hash_data_subject_id(data_subject_identifier)
        
        request_id = f"dsr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tenant_id}"
        request = DataSubjectRequest(
            id=request_id,
            tenant_id=tenant_id,
            data_subject_id=data_subject_id,
            request_type=request_type,
            requested_at=datetime.now(),
            status="pending",
            request_data=request_data
        )
        
        self.data_subject_requests[request_id] = request
        
        if self.enable_audit_logging:
            await self._audit_log("data_subject_request_submitted", {
                "request_id": request_id,
                "tenant_id": tenant_id,
                "request_type": request_type.value
            })
        
        await self._persist_compliance_data()
        logger.info(f"Data subject request submitted: {request_id}")
        return request_id
    
    async def process_right_to_access(self, request_id: str) -> Dict[str, Any]:
        """Process right to access request"""
        request = self.data_subject_requests.get(request_id)
        if not request or request.request_type != DataSubjectRight.ACCESS:
            raise ValueError("Invalid access request")
        
        request.status = "processing"
        request.processed_at = datetime.now()
        
        # Collect all data for this data subject
        data_subject_data = {
            "personal_data": [],
            "processing_activities": [],
            "consents": []
        }
        
        # Find processing records
        for record in self.processing_records.values():
            if (record.tenant_id == request.tenant_id and 
                record.data_subject_id == request.data_subject_id):
                data_subject_data["processing_activities"].append({
                    "id": record.id,
                    "data_category": record.data_category.value,
                    "purpose": record.processing_purpose.value,
                    "processed_at": record.processed_at.isoformat(),
                    "retention_until": record.retention_until.isoformat(),
                    "legal_basis": record.legal_basis
                })
        
        # Find consent records
        for consent in self.consent_records.values():
            if (consent.tenant_id == request.tenant_id and 
                consent.data_subject_id == request.data_subject_id):
                data_subject_data["consents"].append({
                    "id": consent.id,
                    "consent_type": consent.consent_type,
                    "given_at": consent.given_at.isoformat(),
                    "withdrawn_at": consent.withdrawn_at.isoformat() if consent.withdrawn_at else None,
                    "valid": consent.valid,
                    "purpose": consent.purpose.value,
                    "data_categories": [cat.value for cat in consent.data_categories]
                })
        
        # Complete the request
        request.status = "completed"
        request.completed_at = datetime.now()
        request.response_data = data_subject_data
        
        if self.enable_audit_logging:
            await self._audit_log("right_to_access_processed", {
                "request_id": request_id,
                "tenant_id": request.tenant_id,
                "records_found": len(data_subject_data["processing_activities"])
            })
        
        await self._persist_compliance_data()
        return data_subject_data
    
    async def process_right_to_erasure(self, request_id: str) -> Dict[str, Any]:
        """Process right to erasure (right to be forgotten)"""
        request = self.data_subject_requests.get(request_id)
        if not request or request.request_type != DataSubjectRight.ERASURE:
            raise ValueError("Invalid erasure request")
        
        request.status = "processing"
        request.processed_at = datetime.now()
        
        deleted_records = []
        retained_records = []
        
        # Process all records for this data subject
        for record_id, record in self.processing_records.items():
            if (record.tenant_id == request.tenant_id and 
                record.data_subject_id == request.data_subject_id):
                
                # Check if we can delete (consider legal obligations)
                if self._can_delete_record(record):
                    record.deleted = True
                    deleted_records.append(record_id)
                    # Here you would also delete the actual data from your systems
                else:
                    retained_records.append({
                        "record_id": record_id,
                        "reason": "Legal obligation requires retention",
                        "retention_until": record.retention_until.isoformat()
                    })
        
        # Complete the request
        request.status = "completed"
        request.completed_at = datetime.now()
        request.response_data = {
            "deleted_records": deleted_records,
            "retained_records": retained_records,
            "deletion_date": datetime.now().isoformat()
        }
        
        if self.enable_audit_logging:
            await self._audit_log("right_to_erasure_processed", {
                "request_id": request_id,
                "tenant_id": request.tenant_id,
                "deleted_count": len(deleted_records),
                "retained_count": len(retained_records)
            })
        
        await self._persist_compliance_data()
        return request.response_data
    
    async def process_data_portability(self, request_id: str) -> Dict[str, Any]:
        """Process right to data portability"""
        request = self.data_subject_requests.get(request_id)
        if not request or request.request_type != DataSubjectRight.PORTABILITY:
            raise ValueError("Invalid portability request")
        
        request.status = "processing"
        request.processed_at = datetime.now()
        
        # Create portable data export
        portable_data = {
            "export_info": {
                "tenant_id": request.tenant_id,
                "exported_at": datetime.now().isoformat(),
                "format": "JSON",
                "data_subject_id": request.data_subject_id
            },
            "personal_data": {},
            "processing_history": []
        }
        
        # Add processing history
        for record in self.processing_records.values():
            if (record.tenant_id == request.tenant_id and 
                record.data_subject_id == request.data_subject_id and
                not record.deleted):
                portable_data["processing_history"].append({
                    "data_category": record.data_category.value,
                    "purpose": record.processing_purpose.value,
                    "processed_at": record.processed_at.isoformat(),
                    "legal_basis": record.legal_basis
                })
        
        # Mark as exported
        for record in self.processing_records.values():
            if (record.tenant_id == request.tenant_id and 
                record.data_subject_id == request.data_subject_id):
                record.exported = True
        
        request.status = "completed"
        request.completed_at = datetime.now()
        request.response_data = portable_data
        
        if self.enable_audit_logging:
            await self._audit_log("data_portability_processed", {
                "request_id": request_id,
                "tenant_id": request.tenant_id,
                "export_size": len(str(portable_data))
            })
        
        await self._persist_compliance_data()
        return portable_data
    
    def _can_delete_record(self, record: DataProcessingRecord) -> bool:
        """Check if a record can be deleted considering legal obligations"""
        # Swiss law may require retention of certain data types
        if record.processing_purpose == ProcessingPurpose.LEGAL_OBLIGATION:
            return False
        
        # Check if retention period has expired
        if datetime.now() < record.retention_until:
            # Still within retention period, check if early deletion is allowed
            if record.data_category in [DataCategory.FINANCIAL_DATA, DataCategory.HEALTH_DATA]:
                return False  # These typically cannot be deleted early
        
        return True
    
    async def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up data that has exceeded its retention period"""
        now = datetime.now()
        cleanup_stats = {
            "records_deleted": 0,
            "consents_expired": 0,
            "requests_archived": 0
        }
        
        # Clean up expired processing records
        expired_records = []
        for record_id, record in self.processing_records.items():
            if now > record.retention_until and not record.deleted:
                record.deleted = True
                expired_records.append(record_id)
                cleanup_stats["records_deleted"] += 1
        
        # Archive old completed requests (older than 1 year)
        archive_cutoff = now - timedelta(days=365)
        archived_requests = []
        for request_id, request in self.data_subject_requests.items():
            if (request.completed_at and 
                request.completed_at < archive_cutoff):
                archived_requests.append(request_id)
                cleanup_stats["requests_archived"] += 1
        
        # Remove archived requests
        for request_id in archived_requests:
            del self.data_subject_requests[request_id]
        
        if self.enable_audit_logging:
            await self._audit_log("data_cleanup_performed", cleanup_stats)
        
        await self._persist_compliance_data()
        logger.info(f"Data cleanup completed: {cleanup_stats}")
        return cleanup_stats
    
    async def get_compliance_report(self, tenant_id: str) -> Dict[str, Any]:
        """Generate compliance report for a tenant"""
        report = {
            "tenant_id": tenant_id,
            "report_date": datetime.now().isoformat(),
            "data_residency": self.data_residency_region,
            "processing_activities": {
                "total": 0,
                "by_category": {},
                "by_purpose": {},
                "active": 0,
                "deleted": 0
            },
            "consents": {
                "total": 0,
                "active": 0,
                "withdrawn": 0
            },
            "data_subject_requests": {
                "total": 0,
                "by_type": {},
                "pending": 0,
                "completed": 0
            },
            "retention_compliance": {
                "expiring_soon": [],
                "overdue": []
            }
        }
        
        # Analyze processing records
        for record in self.processing_records.values():
            if record.tenant_id == tenant_id:
                report["processing_activities"]["total"] += 1
                
                # By category
                category = record.data_category.value
                report["processing_activities"]["by_category"][category] = \
                    report["processing_activities"]["by_category"].get(category, 0) + 1
                
                # By purpose
                purpose = record.processing_purpose.value
                report["processing_activities"]["by_purpose"][purpose] = \
                    report["processing_activities"]["by_purpose"].get(purpose, 0) + 1
                
                # Status
                if record.deleted:
                    report["processing_activities"]["deleted"] += 1
                else:
                    report["processing_activities"]["active"] += 1
                
                # Retention compliance
                days_until_expiry = (record.retention_until - datetime.now()).days
                if days_until_expiry <= 30 and days_until_expiry > 0:
                    report["retention_compliance"]["expiring_soon"].append({
                        "record_id": record.id,
                        "days_remaining": days_until_expiry,
                        "data_category": category
                    })
                elif days_until_expiry < 0:
                    report["retention_compliance"]["overdue"].append({
                        "record_id": record.id,
                        "days_overdue": abs(days_until_expiry),
                        "data_category": category
                    })
        
        # Analyze consents
        for consent in self.consent_records.values():
            if consent.tenant_id == tenant_id:
                report["consents"]["total"] += 1
                if consent.valid:
                    report["consents"]["active"] += 1
                else:
                    report["consents"]["withdrawn"] += 1
        
        # Analyze data subject requests
        for request in self.data_subject_requests.values():
            if request.tenant_id == tenant_id:
                report["data_subject_requests"]["total"] += 1
                
                request_type = request.request_type.value
                report["data_subject_requests"]["by_type"][request_type] = \
                    report["data_subject_requests"]["by_type"].get(request_type, 0) + 1
                
                if request.status == "completed":
                    report["data_subject_requests"]["completed"] += 1
                elif request.status == "pending":
                    report["data_subject_requests"]["pending"] += 1
        
        return report
    
    async def _audit_log(self, action: str, data: Dict[str, Any]):
        """Log compliance-related actions for audit purposes"""
        if not self.enable_audit_logging:
            return
        
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "data": data,
            "service": "swiss_data_protection"
        }
        
        # Write to audit log file
        audit_file = self.storage_path / "audit.jsonl"
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry) + "\n")
    
    async def _persist_compliance_data(self):
        """Persist compliance data to storage"""
        try:
            # Convert records to serializable format
            data = {
                "processing_records": {
                    record_id: self._record_to_dict(record)
                    for record_id, record in self.processing_records.items()
                },
                "consent_records": {
                    consent_id: self._consent_to_dict(consent)
                    for consent_id, consent in self.consent_records.items()
                },
                "data_subject_requests": {
                    request_id: self._request_to_dict(request)
                    for request_id, request in self.data_subject_requests.items()
                },
                "last_updated": datetime.now().isoformat()
            }
            
            # Write to file
            compliance_file = self.storage_path / "compliance_data.json"
            with open(compliance_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to persist compliance data: {e}")
    
    def _load_compliance_data(self):
        """Load compliance data from storage"""
        try:
            compliance_file = self.storage_path / "compliance_data.json"
            if not compliance_file.exists():
                return
            
            with open(compliance_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Load processing records
            for record_id, record_data in data.get("processing_records", {}).items():
                self.processing_records[record_id] = self._dict_to_record(record_data)
            
            # Load consent records
            for consent_id, consent_data in data.get("consent_records", {}).items():
                self.consent_records[consent_id] = self._dict_to_consent(consent_data)
            
            # Load data subject requests
            for request_id, request_data in data.get("data_subject_requests", {}).items():
                self.data_subject_requests[request_id] = self._dict_to_request(request_data)
            
            logger.info(f"Loaded compliance data: {len(self.processing_records)} records, "
                       f"{len(self.consent_records)} consents, {len(self.data_subject_requests)} requests")
            
        except Exception as e:
            logger.error(f"Failed to load compliance data: {e}")
    
    def _record_to_dict(self, record: DataProcessingRecord) -> Dict[str, Any]:
        """Convert DataProcessingRecord to dictionary"""
        data = asdict(record)
        data["data_category"] = data["data_category"].value
        data["processing_purpose"] = data["processing_purpose"].value
        data["processed_at"] = data["processed_at"].isoformat()
        data["retention_until"] = data["retention_until"].isoformat()
        return data
    
    def _dict_to_record(self, data: Dict[str, Any]) -> DataProcessingRecord:
        """Convert dictionary to DataProcessingRecord"""
        data["data_category"] = DataCategory(data["data_category"])
        data["processing_purpose"] = ProcessingPurpose(data["processing_purpose"])
        data["processed_at"] = datetime.fromisoformat(data["processed_at"])
        data["retention_until"] = datetime.fromisoformat(data["retention_until"])
        return DataProcessingRecord(**data)
    
    def _consent_to_dict(self, consent: ConsentRecord) -> Dict[str, Any]:
        """Convert ConsentRecord to dictionary"""
        data = asdict(consent)
        data["purpose"] = data["purpose"].value
        data["data_categories"] = [cat.value for cat in data["data_categories"]]
        data["given_at"] = data["given_at"].isoformat()
        if data["withdrawn_at"]:
            data["withdrawn_at"] = data["withdrawn_at"].isoformat()
        return data
    
    def _dict_to_consent(self, data: Dict[str, Any]) -> ConsentRecord:
        """Convert dictionary to ConsentRecord"""
        data["purpose"] = ProcessingPurpose(data["purpose"])
        data["data_categories"] = [DataCategory(cat) for cat in data["data_categories"]]
        data["given_at"] = datetime.fromisoformat(data["given_at"])
        if data["withdrawn_at"]:
            data["withdrawn_at"] = datetime.fromisoformat(data["withdrawn_at"])
        return ConsentRecord(**data)
    
    def _request_to_dict(self, request: DataSubjectRequest) -> Dict[str, Any]:
        """Convert DataSubjectRequest to dictionary"""
        data = asdict(request)
        data["request_type"] = data["request_type"].value
        data["requested_at"] = data["requested_at"].isoformat()
        if data["processed_at"]:
            data["processed_at"] = data["processed_at"].isoformat()
        if data["completed_at"]:
            data["completed_at"] = data["completed_at"].isoformat()
        return data
    
    def _dict_to_request(self, data: Dict[str, Any]) -> DataSubjectRequest:
        """Convert dictionary to DataSubjectRequest"""
        data["request_type"] = DataSubjectRight(data["request_type"])
        data["requested_at"] = datetime.fromisoformat(data["requested_at"])
        if data["processed_at"]:
            data["processed_at"] = datetime.fromisoformat(data["processed_at"])
        if data["completed_at"]:
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])
        return DataSubjectRequest(**data)

# Global service instance
_compliance_service: Optional[SwissDataProtectionService] = None

def get_compliance_service() -> SwissDataProtectionService:
    """Get global compliance service"""
    global _compliance_service
    if _compliance_service is None:
        _compliance_service = SwissDataProtectionService()
    return _compliance_service

def initialize_compliance_service(
    storage_path: str = "data/compliance",
    enable_audit_logging: bool = True,
    data_residency_region: str = "CH"
) -> SwissDataProtectionService:
    """Initialize global compliance service"""
    global _compliance_service
    _compliance_service = SwissDataProtectionService(
        storage_path=storage_path,
        enable_audit_logging=enable_audit_logging,
        data_residency_region=data_residency_region
    )
    return _compliance_service