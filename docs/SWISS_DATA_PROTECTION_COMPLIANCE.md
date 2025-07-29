# Swiss Data Protection Compliance Guide

This guide covers compliance with Swiss Data Protection Act (DSG) and GDPR requirements for the RAG System.

## 🇨🇭 Swiss Data Protection Overview

### Legal Framework
- **Swiss Data Protection Act (DSG)** - Revised law effective September 1, 2023
- **GDPR Compatibility** - DSG is designed to be equivalent to EU GDPR
- **Data Residency** - Data processing occurs within Switzerland
- **Cross-Border Transfers** - Regulated transfers to adequate countries

### Key Principles
1. **Lawfulness** - Processing must have legal basis
2. **Purpose Limitation** - Data used only for stated purposes
3. **Data Minimization** - Only necessary data is processed
4. **Accuracy** - Data must be accurate and up-to-date
5. **Storage Limitation** - Data retained only as long as necessary
6. **Integrity & Confidentiality** - Appropriate security measures

## 📋 Data Categories & Processing

### Personal Data Categories

| Category | Examples | Retention Period | Special Requirements |
|----------|----------|------------------|---------------------|
| **Personal Identifiers** | Names, email, phone | 7 years | Standard protection |
| **Financial Data** | Bank details, payments | 10 years | Enhanced security |
| **Health Data** | Medical records | 7 years | Special category data |
| **Behavioral Data** | Usage patterns, preferences | 1 year | Anonymization required |
| **Technical Data** | IP addresses, device info | 3 months | Pseudonymization |
| **Communication Data** | Messages, documents | 7 years | Encryption required |

### Processing Purposes & Legal Basis

```python
# Legal bases for processing
LEGAL_BASES = {
    "contract": "Processing necessary for contract performance",
    "legal_obligation": "Processing required by Swiss law",
    "legitimate_interest": "Processing for legitimate business interests",
    "consent": "Data subject has given explicit consent",
    "public_interest": "Processing for public task",
    "vital_interest": "Processing to protect vital interests"
}
```

## 🔐 Implementation in RAG System

### 1. Data Processing Recording

Every data processing activity must be recorded:

```python
from core.services.compliance_service import get_compliance_service

# Record document upload
await compliance_service.record_data_processing(
    tenant_id="company-123",
    data_category="communication_data",
    processing_purpose="legitimate_interest",
    data_subject_identifier="user@company.com",
    data_source="document_upload",
    legal_basis="legitimate_interest"
)
```

### 2. Consent Management

```python
# Record consent for processing
consent_id = await compliance_service.record_consent(
    tenant_id="company-123",
    data_subject_identifier="user@company.com",
    consent_type="document_processing",
    purpose="legitimate_interest",
    data_categories=["communication_data", "technical_data"]
)

# Withdraw consent
await compliance_service.withdraw_consent(consent_id)
```

### 3. Data Subject Rights

#### Right to Access
```python
# Submit access request
request_id = await compliance_service.submit_data_subject_request(
    tenant_id="company-123",
    data_subject_identifier="user@company.com",
    request_type="access",
    request_data={"scope": "all_data"}
)

# Process request
data = await compliance_service.process_right_to_access(request_id)
```

#### Right to Erasure (Right to be Forgotten)
```python
# Submit erasure request
request_id = await compliance_service.submit_data_subject_request(
    tenant_id="company-123",
    data_subject_identifier="user@company.com",
    request_type="erasure",
    request_data={"reason": "withdrawal_of_consent"}
)

# Process erasure
result = await compliance_service.process_right_to_erasure(request_id)
```

#### Data Portability
```python
# Export data in portable format
request_id = await compliance_service.submit_data_subject_request(
    tenant_id="company-123",
    data_subject_identifier="user@company.com",
    request_type="portability",
    request_data={"format": "json"}
)

portable_data = await compliance_service.process_data_portability(request_id)
```

## 🔒 Security & Privacy by Design

### 1. Data Pseudonymization
```python
# User identifiers are automatically hashed
hashed_id = compliance_service.hash_data_subject_id("user@company.com")
# Results in: "a1b2c3d4e5f6..." (SHA-256 hash)
```

### 2. Encryption at Rest
```yaml
# Environment configuration
ENCRYPTION_ENABLED=true
ENCRYPTION_KEY_FILE=data/encryption.key
TENANT_SPECIFIC_ENCRYPTION=true
```

### 3. Access Controls
- **Role-based access** to compliance data
- **Audit logging** for all compliance operations
- **Multi-tenant isolation** prevents cross-tenant access

### 4. Data Retention
Automatic cleanup of expired data based on Swiss law requirements:
```python
# Automated daily cleanup
cleanup_stats = await compliance_service.cleanup_expired_data()
```

## 📊 Compliance Monitoring

### 1. Compliance Dashboard
Access via API: `GET /api/v1/compliance/tenants/{tenant_id}/report`

Returns:
- Processing activities breakdown
- Consent status summary
- Data subject requests status
- Retention compliance alerts

### 2. Audit Trail
All compliance operations are logged:
```json
{
  "timestamp": "2025-01-29T10:30:00Z",
  "action": "data_processing_recorded",
  "data": {
    "record_id": "proc_20250129_103000_company-123",
    "tenant_id": "company-123",
    "data_category": "communication_data",
    "legal_basis": "legitimate_interest"
  }
}
```

### 3. Automated Alerts
- **Retention expiry warnings** (30 days before)
- **Overdue data cleanup** alerts
- **Unusual processing activity** notifications

## 🎯 API Endpoints

### Processing Records
```http
POST /api/v1/compliance/processing/record
GET /api/v1/compliance/tenants/{tenant_id}/report
```

### Consent Management
```http
POST /api/v1/compliance/consent/record
POST /api/v1/compliance/consent/{consent_id}/withdraw
```

### Data Subject Rights
```http
POST /api/v1/compliance/data-subject-requests/submit
POST /api/v1/compliance/data-subject-requests/{request_id}/process
```

### Maintenance
```http
POST /api/v1/compliance/cleanup/expired-data
GET /api/v1/compliance/health
```

## 📝 Compliance Checklist

### ✅ Technical Implementation
- ✅ **Data processing recording** - All activities logged
- ✅ **Consent management** - Full consent lifecycle
- ✅ **Data subject rights** - Access, erasure, portability
- ✅ **Pseudonymization** - User identifiers hashed
- ✅ **Encryption at rest** - Tenant-specific encryption
- ✅ **Audit logging** - Complete audit trail
- ✅ **Data retention** - Automatic cleanup
- ✅ **Multi-tenancy** - Data isolation

### ✅ Legal Compliance
- ✅ **Swiss data residency** - Processing within Switzerland
- ✅ **Legal basis documentation** - All processing justified
- ✅ **Retention periods** - Swiss law compliant
- ✅ **Data subject rights** - All GDPR/DSG rights implemented
- ✅ **Privacy by design** - Built-in privacy protection
- ✅ **Security measures** - Appropriate technical safeguards

### 📋 Operational Requirements
- ✅ **Data Protection Officer** - Designate if required
- ✅ **Privacy policy** - Document processing activities
- ✅ **Staff training** - Data protection awareness
- ✅ **Impact assessments** - For high-risk processing
- ✅ **Breach procedures** - Incident response plan
- ✅ **Regular audits** - Compliance monitoring

## 🚨 Incident Response

### Data Breach Procedure
1. **Detection** - Automated alerts and monitoring
2. **Assessment** - Determine breach severity
3. **Notification** - 72-hour authority notification if required
4. **Documentation** - Record in compliance system
5. **Mitigation** - Implement corrective measures

### Emergency Data Deletion
```python
# Emergency erasure for data subject
await compliance_service.process_right_to_erasure(request_id)

# Verify deletion
report = await compliance_service.get_compliance_report(tenant_id)
```

## 📞 Support & Resources

### Swiss Data Protection Authority
- **Website**: https://www.edoeb.admin.ch/
- **Guidance**: DSG implementation guidelines
- **Reporting**: Data breach notification procedures

### GDPR Resources
- **Official text**: https://gdpr-info.eu/
- **Guidance**: European Data Protection Board guidelines
- **Templates**: Privacy policies and consent forms

### Internal Resources
- **Compliance API**: `/api/v1/compliance/`
- **Health checks**: `/api/v1/compliance/health`
- **Documentation**: This guide and API documentation
- **Audit logs**: `data/compliance/audit.jsonl`

---

**⚖️ Legal Disclaimer**: This implementation provides technical tools for compliance but does not constitute legal advice. Consult with qualified data protection lawyers for specific legal requirements in your jurisdiction.

**🔄 Last Updated**: January 2025  
**📝 Version**: 1.0  
**📍 Jurisdiction**: Switzerland (DSG) + EU (GDPR)