"""
Compliance API Router
Provides endpoints for GDPR/DSG compliance operations
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from ..services.compliance_service import (
    DataCategory,
    DataSubjectRight,
    ProcessingPurpose,
    get_compliance_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/compliance", tags=["compliance"])


# Request/Response Models
class DataProcessingRequest(BaseModel):
    tenant_id: str
    data_category: str = Field(..., description="Category of data being processed")
    processing_purpose: str = Field(..., description="Purpose for processing")
    data_subject_identifier: Optional[str] = Field(
        None, description="Identifier for the data subject"
    )
    data_source: str = Field("api", description="Source of the data")
    legal_basis: str = Field(
        "legitimate_interest", description="Legal basis for processing"
    )
    custom_retention_days: Optional[int] = Field(
        None, description="Custom retention period in days"
    )


class ConsentRequest(BaseModel):
    tenant_id: str
    data_subject_identifier: str
    consent_type: str
    processing_purpose: str
    data_categories: List[str]


class DataSubjectRequestModel(BaseModel):
    tenant_id: str
    data_subject_identifier: str
    request_type: str = Field(
        ..., description="Type of request: access, erasure, portability, etc."
    )
    request_data: Dict[str, Any] = Field(default_factory=dict)


class ComplianceResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


def validate_data_category(category: str) -> DataCategory:
    """Validate and convert data category string"""
    try:
        return DataCategory(category.lower())
    except ValueError:
        valid_categories = [cat.value for cat in DataCategory]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data category '{category}'. Valid options: {valid_categories}",
        )


def validate_processing_purpose(purpose: str) -> ProcessingPurpose:
    """Validate and convert processing purpose string"""
    try:
        return ProcessingPurpose(purpose.lower())
    except ValueError:
        valid_purposes = [purpose.value for purpose in ProcessingPurpose]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid processing purpose '{purpose}'. Valid options: {valid_purposes}",
        )


def validate_data_subject_right(right: str) -> DataSubjectRight:
    """Validate and convert data subject right string"""
    try:
        return DataSubjectRight(right.lower())
    except ValueError:
        valid_rights = [right.value for right in DataSubjectRight]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data subject right '{right}'. Valid options: {valid_rights}",
        )


@router.post("/processing/record")
async def record_data_processing(request: DataProcessingRequest) -> ComplianceResponse:
    """
    Record a data processing activity

    This endpoint should be called whenever personal data is processed
    to maintain compliance with GDPR/DSG requirements.
    """
    try:
        compliance_service = get_compliance_service()

        # Validate inputs
        data_category = validate_data_category(request.data_category)
        processing_purpose = validate_processing_purpose(request.processing_purpose)

        # Record the processing activity
        record_id = await compliance_service.record_data_processing(
            tenant_id=request.tenant_id,
            data_category=data_category,
            processing_purpose=processing_purpose,
            data_subject_identifier=request.data_subject_identifier,
            data_source=request.data_source,
            legal_basis=request.legal_basis,
            custom_retention_days=request.custom_retention_days,
        )

        logger.info(f"Data processing recorded: {record_id}")

        return ComplianceResponse(
            success=True,
            message="Data processing activity recorded successfully",
            data={"record_id": record_id},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to record data processing: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to record data processing: {str(e)}"
        )


@router.post("/consent/record")
async def record_consent(request: ConsentRequest) -> ComplianceResponse:
    """
    Record data subject consent

    Use this endpoint when a data subject provides explicit consent
    for data processing activities.
    """
    try:
        compliance_service = get_compliance_service()

        # Validate inputs
        processing_purpose = validate_processing_purpose(request.processing_purpose)
        data_categories = [
            validate_data_category(cat) for cat in request.data_categories
        ]

        # Record consent
        consent_id = await compliance_service.record_consent(
            tenant_id=request.tenant_id,
            data_subject_identifier=request.data_subject_identifier,
            consent_type=request.consent_type,
            purpose=processing_purpose,
            data_categories=data_categories,
        )

        logger.info(f"Consent recorded: {consent_id}")

        return ComplianceResponse(
            success=True,
            message="Consent recorded successfully",
            data={"consent_id": consent_id},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to record consent: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to record consent: {str(e)}"
        )


@router.post("/consent/{consent_id}/withdraw")
async def withdraw_consent(consent_id: str) -> ComplianceResponse:
    """
    Withdraw previously given consent

    This endpoint allows data subjects to withdraw their consent,
    which will stop processing based on consent.
    """
    try:
        compliance_service = get_compliance_service()

        success = await compliance_service.withdraw_consent(consent_id)

        if not success:
            raise HTTPException(
                status_code=404, detail="Consent not found or already withdrawn"
            )

        logger.info(f"Consent withdrawn: {consent_id}")

        return ComplianceResponse(
            success=True,
            message="Consent withdrawn successfully",
            data={"consent_id": consent_id, "withdrawn_at": datetime.now().isoformat()},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to withdraw consent: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to withdraw consent: {str(e)}"
        )


@router.post("/data-subject-requests/submit")
async def submit_data_subject_request(
    request: DataSubjectRequestModel,
) -> ComplianceResponse:
    """
    Submit a data subject rights request

    Supports all GDPR/DSG rights including access, erasure, portability,
    rectification, and restriction of processing.
    """
    try:
        compliance_service = get_compliance_service()

        # Validate request type
        request_type = validate_data_subject_right(request.request_type)

        # Submit the request
        request_id = await compliance_service.submit_data_subject_request(
            tenant_id=request.tenant_id,
            data_subject_identifier=request.data_subject_identifier,
            request_type=request_type,
            request_data=request.request_data,
        )

        logger.info(f"Data subject request submitted: {request_id}")

        return ComplianceResponse(
            success=True,
            message="Data subject request submitted successfully",
            data={
                "request_id": request_id,
                "request_type": request_type.value,
                "status": "pending",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit data subject request: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to submit data subject request: {str(e)}"
        )


@router.post("/data-subject-requests/{request_id}/process")
async def process_data_subject_request(request_id: str) -> ComplianceResponse:
    """
    Process a data subject rights request

    This endpoint processes the request and returns the appropriate response
    based on the request type (access, erasure, portability).
    """
    try:
        compliance_service = get_compliance_service()

        # Get the request to determine type
        request = compliance_service.data_subject_requests.get(request_id)
        if not request:
            raise HTTPException(status_code=404, detail="Request not found")

        # Process based on request type
        if request.request_type == DataSubjectRight.ACCESS:
            result = await compliance_service.process_right_to_access(request_id)
            message = "Access request processed successfully"
        elif request.request_type == DataSubjectRight.ERASURE:
            result = await compliance_service.process_right_to_erasure(request_id)
            message = "Erasure request processed successfully"
        elif request.request_type == DataSubjectRight.PORTABILITY:
            result = await compliance_service.process_data_portability(request_id)
            message = "Data portability request processed successfully"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Processing not implemented for request type: {request.request_type.value}",
            )

        logger.info(f"Data subject request processed: {request_id}")

        return ComplianceResponse(
            success=True,
            message=message,
            data={
                "request_id": request_id,
                "request_type": request.request_type.value,
                "result": result,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process data subject request: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process data subject request: {str(e)}"
        )


@router.get("/tenants/{tenant_id}/report")
async def get_compliance_report(tenant_id: str) -> ComplianceResponse:
    """
    Get compliance report for a tenant

    Returns a comprehensive report of all data processing activities,
    consents, and data subject requests for the specified tenant.
    """
    try:
        compliance_service = get_compliance_service()

        report = await compliance_service.get_compliance_report(tenant_id)

        return ComplianceResponse(
            success=True,
            message="Compliance report generated successfully",
            data=report,
        )

    except Exception as e:
        logger.error(f"Failed to generate compliance report: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate compliance report: {str(e)}"
        )


@router.post("/cleanup/expired-data")
async def cleanup_expired_data(background_tasks: BackgroundTasks) -> ComplianceResponse:
    """
    Clean up expired data

    Removes data that has exceeded its retention period according to
    Swiss data protection and GDPR requirements.
    """
    try:
        compliance_service = get_compliance_service()

        # Run cleanup in background
        background_tasks.add_task(compliance_service.cleanup_expired_data)

        return ComplianceResponse(
            success=True,
            message="Data cleanup initiated in background",
            data={"status": "initiated"},
        )

    except Exception as e:
        logger.error(f"Failed to initiate data cleanup: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to initiate data cleanup: {str(e)}"
        )


@router.get("/health")
async def compliance_health_check() -> ComplianceResponse:
    """
    Health check for compliance service

    Returns the current status of the compliance service and basic statistics.
    """
    try:
        compliance_service = get_compliance_service()

        # Get basic statistics
        total_records = len(compliance_service.processing_records)
        total_consents = len(compliance_service.consent_records)
        total_requests = len(compliance_service.data_subject_requests)

        health_data = {
            "service": "swiss_data_protection",
            "status": "healthy",
            "data_residency": compliance_service.data_residency_region,
            "statistics": {
                "processing_records": total_records,
                "consent_records": total_consents,
                "data_subject_requests": total_requests,
            },
        }

        return ComplianceResponse(
            success=True, message="Compliance service is healthy", data=health_data
        )

    except Exception as e:
        logger.error(f"Compliance health check failed: {e}")
        return ComplianceResponse(
            success=False,
            message=f"Compliance service unhealthy: {str(e)}",
            data={"service": "swiss_data_protection", "status": "unhealthy"},
        )


@router.get("/info/categories")
async def get_data_categories() -> Dict[str, List[str]]:
    """Get list of available data categories"""
    return {
        "data_categories": [cat.value for cat in DataCategory],
        "processing_purposes": [purpose.value for purpose in ProcessingPurpose],
        "data_subject_rights": [right.value for right in DataSubjectRight],
    }
