"""
Job management endpoints.

Async jobs are created by content/distribute endpoints and polled here.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.auth import verify_api_key
from src.api.job_manager import get_job_manager
from src.api.models import (
    JobStatus,
    JobStatusResponse,
    JobResultResponse,
)

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, _: str = Depends(verify_api_key)):
    """Get current status of an async job."""
    mgr = get_job_manager()
    job = mgr.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        job_type=job.job_type,
        progress=job.progress,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        result=job.result,
        error=job.error,
        params=job.params,
    )


@router.get("/{job_id}/result", response_model=JobResultResponse)
async def get_job_result(job_id: str, _: str = Depends(verify_api_key)):
    """Get full result of a completed job, including artifacts."""
    mgr = get_job_manager()
    job = mgr.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
        raise HTTPException(
            status_code=409,
            detail=f"Job is still {job.status.value}. Poll /jobs/{job_id} for status.",
        )

    result = job.result or {}
    artifacts = result.pop("artifacts", []) if isinstance(result, dict) else []
    metrics = result.pop("metrics", {}) if isinstance(result, dict) else {}

    return JobResultResponse(
        job_id=job.id,
        status=job.status,
        result=result,
        artifacts=artifacts,
        metrics=metrics,
    )


@router.delete("/{job_id}")
async def cancel_job(job_id: str, _: str = Depends(verify_api_key)):
    """Cancel a queued job. Running jobs cannot be cancelled."""
    mgr = get_job_manager()
    job = mgr.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if not mgr.cancel_job(job_id):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel job in state: {job.status.value}",
        )

    return {"job_id": job_id, "status": "cancelled"}


@router.get("/", response_model=list)
async def list_jobs(
    status: Optional[JobStatus] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    _: str = Depends(verify_api_key),
):
    """List jobs, optionally filtered by status."""
    mgr = get_job_manager()
    jobs = mgr.list_jobs(status=status, limit=limit)
    return [
        {
            "job_id": j.id,
            "job_type": j.job_type,
            "status": j.status.value,
            "progress": j.progress,
            "created_at": j.created_at.isoformat(),
            "completed_at": j.completed_at.isoformat() if j.completed_at else None,
        }
        for j in jobs
    ]
