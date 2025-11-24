from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from any_llm.gateway.auth import verify_jwt_or_api_key_or_master
from any_llm.gateway.db import APIKey, Budget, CaretUser, UsageLog, User, get_db

router = APIRouter(prefix="/v1/profile", tags=["profile"])


class BudgetInfo(BaseModel):
    """예산 정보."""

    budget_id: str | None
    max_budget: float | None
    budget_duration_sec: int | None
    spend: float
    budget_started_at: str | None
    next_budget_reset_at: str | None


class ProfileInfo(BaseModel):
    """사용자/소셜 통합 프로필."""

    user_id: str
    provider: str | None
    provider_user_id: str | None
    alias: str | None
    name: str | None
    email: str | None
    avatar_url: str | None
    blocked: bool
    metadata: dict
    caret_metadata: dict
    last_login_at: str | None


class UsageWindow(BaseModel):
    """기간별 사용량 합계."""

    requests: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float


class UsageLogItem(BaseModel):
    """최근 사용 로그 일부."""

    id: str
    timestamp: str
    model: str
    provider: str | None
    endpoint: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost: float | None
    status: str
    error_message: str | None


class ProfileResponse(BaseModel):
    """프로필 응답."""

    profile: ProfileInfo
    budget: BudgetInfo | None
    usage: dict[str, UsageWindow]
    recent_usage: list[UsageLogItem]


class UsageBucket(BaseModel):
    """그룹별 집계."""

    period: str
    requests: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float


class UsageBucketsResponse(BaseModel):
    """기간별 집계 응답."""

    group_by: str
    start: str
    end: str
    buckets: list[UsageBucket]
    total: UsageBucket


class KeySummary(BaseModel):
    """API 키 메타 요약(평문 키 미노출)."""

    id: str
    key_name: str | None
    user_id: str | None
    created_at: str
    last_used_at: str | None
    expires_at: str | None
    is_active: bool
    metadata: dict


def _now_naive() -> datetime:
    """UTC now without tzinfo (DB 저장 방식과 동일)."""
    return datetime.now(UTC).replace(tzinfo=None)


def _ensure_naive(dt: datetime) -> datetime:
    """Ensure datetime is naive UTC."""
    if dt.tzinfo:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _resolve_target_user(auth_result: tuple[APIKey | None, bool, str | None], user_param: str | None) -> str:
    """Resolve target user_id from auth and optional query."""
    api_key, is_master, resolved_user_id = auth_result
    if is_master:
        if not user_param:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="When using master key, 'user' query parameter is required",
            )
        return user_param

    target_user_id = resolved_user_id or (api_key.user_id if api_key else None)
    if not target_user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not resolved")
    return target_user_id


def _aggregate_usage(db: Session, user_id: str, since: datetime) -> UsageWindow:
    """기간별 사용량 합계."""
    requests_count, prompt_sum, completion_sum, total_sum, cost_sum = (
        db.query(
            func.count(UsageLog.id),
            func.coalesce(func.sum(UsageLog.prompt_tokens), 0),
            func.coalesce(func.sum(UsageLog.completion_tokens), 0),
            func.coalesce(func.sum(UsageLog.total_tokens), 0),
            func.coalesce(func.sum(UsageLog.cost), 0.0),
        )
        .filter(UsageLog.user_id == user_id, UsageLog.timestamp >= since)
        .one()
    )

    return UsageWindow(
        requests=int(requests_count),
        prompt_tokens=int(prompt_sum or 0),
        completion_tokens=int(completion_sum or 0),
        total_tokens=int(total_sum or 0),
        cost=float(cost_sum or 0.0),
    )


def _recent_usage(db: Session, user_id: str, limit: int) -> list[UsageLogItem]:
    """최근 사용 로그."""
    logs = (
        db.query(UsageLog)
        .filter(UsageLog.user_id == user_id)
        .order_by(UsageLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        UsageLogItem(
            id=log.id,
            timestamp=log.timestamp.isoformat(),
            model=log.model,
            provider=log.provider,
            endpoint=log.endpoint,
            prompt_tokens=log.prompt_tokens,
            completion_tokens=log.completion_tokens,
            total_tokens=log.total_tokens,
            cost=log.cost,
            status=log.status,
            error_message=log.error_message,
        )
        for log in logs
    ]


@router.get("")
async def get_profile(
    auth_result: Annotated[tuple[APIKey | None, bool, str | None], Depends(verify_jwt_or_api_key_or_master)],
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[str | None, Query(default=None, description="마스터 키 사용 시 조회할 user_id")],
    recent_limit: Annotated[int, Query(default=10, ge=0, le=100, description="최근 사용 로그 개수")],
) -> ProfileResponse:
    """프로필 + 예산 + 사용량 집계 반환."""
    target_user_id = _resolve_target_user(auth_result, user)
    user_obj = db.query(User).filter(User.user_id == target_user_id).first()
    if not user_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User '{target_user_id}' not found")

    caret = db.query(CaretUser).filter(CaretUser.user_id == target_user_id).first()
    budget = db.query(Budget).filter(Budget.budget_id == user_obj.budget_id).first() if user_obj.budget_id else None

    now = _now_naive()
    usage_windows = {
        "last_24h": _aggregate_usage(db, target_user_id, now - timedelta(hours=24)),
        "last_7d": _aggregate_usage(db, target_user_id, now - timedelta(days=7)),
        "last_30d": _aggregate_usage(db, target_user_id, now - timedelta(days=30)),
    }

    recent_logs = _recent_usage(db, target_user_id, recent_limit)

    profile = ProfileInfo(
        user_id=user_obj.user_id,
        provider=caret.provider if caret else None,
        provider_user_id=caret.provider_user_id if caret else None,
        alias=user_obj.alias,
        name=caret.name if caret else user_obj.alias,
        email=caret.email if caret else None,
        avatar_url=caret.avatar_url if caret else None,
        blocked=bool(user_obj.blocked),
        metadata=dict(user_obj.metadata_) if user_obj.metadata_ else {},
        caret_metadata=dict(caret.metadata_) if caret and caret.metadata_ else {},
        last_login_at=caret.last_login_at.isoformat() if caret and caret.last_login_at else None,
    )

    budget_info = (
        BudgetInfo(
            budget_id=budget.budget_id,
            max_budget=budget.max_budget,
            budget_duration_sec=budget.budget_duration_sec,
            spend=float(user_obj.spend),
            budget_started_at=user_obj.budget_started_at.isoformat() if user_obj.budget_started_at else None,
            next_budget_reset_at=user_obj.next_budget_reset_at.isoformat() if user_obj.next_budget_reset_at else None,
        )
        if budget
        else None
    )

    return ProfileResponse(
        profile=profile,
        budget=budget_info,
        usage=usage_windows,
        recent_usage=recent_logs,
    )


@router.get("/usage")
async def get_profile_usage(
    auth_result: Annotated[tuple[APIKey | None, bool, str | None], Depends(verify_jwt_or_api_key_or_master)],
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[str | None, Query(default=None, description="마스터 키 사용 시 조회할 user_id")],
    start: Annotated[datetime | None, Query(default=None, description="집계 시작 시각(ISO). 기본: now-30d")],
    end: Annotated[datetime | None, Query(default=None, description="집계 종료 시각(ISO). 기본: now")],
    group_by: Annotated[Literal["day", "week", "total"], Query(default="day", description="집계 단위")] = "day",
) -> UsageBucketsResponse:
    """사용량 집계(기간별)."""
    target_user_id = _resolve_target_user(auth_result, user)

    now = _now_naive()
    start_dt = _ensure_naive(start) if start else now - timedelta(days=30)
    end_dt = _ensure_naive(end) if end else now
    if start_dt > end_dt:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="start must be before end")

    total = _aggregate_range(db, target_user_id, start_dt, end_dt)

    if group_by == "total":
        buckets = [
            UsageBucket(
                period="total",
                requests=total.requests,
                prompt_tokens=total.prompt_tokens,
                completion_tokens=total.completion_tokens,
                total_tokens=total.total_tokens,
                cost=total.cost,
            )
        ]
    else:
        truncate_unit = "day" if group_by == "day" else "week"
        rows = (
            db.query(
                func.date_trunc(truncate_unit, UsageLog.timestamp).label("period"),
                func.count(UsageLog.id),
                func.coalesce(func.sum(UsageLog.prompt_tokens), 0),
                func.coalesce(func.sum(UsageLog.completion_tokens), 0),
                func.coalesce(func.sum(UsageLog.total_tokens), 0),
                func.coalesce(func.sum(UsageLog.cost), 0.0),
            )
            .filter(UsageLog.user_id == target_user_id, UsageLog.timestamp >= start_dt, UsageLog.timestamp <= end_dt)
            .group_by("period")
            .order_by("period")
            .all()
        )

        buckets = [
            UsageBucket(
                period=row[0].isoformat(),
                requests=int(row[1]),
                prompt_tokens=int(row[2] or 0),
                completion_tokens=int(row[3] or 0),
                total_tokens=int(row[4] or 0),
                cost=float(row[5] or 0.0),
            )
            for row in rows
        ]

    return UsageBucketsResponse(
        group_by=group_by,
        start=start_dt.isoformat(),
        end=end_dt.isoformat(),
        buckets=buckets,
        total=UsageBucket(
            period="total",
            requests=total.requests,
            prompt_tokens=total.prompt_tokens,
            completion_tokens=total.completion_tokens,
            total_tokens=total.total_tokens,
            cost=total.cost,
        ),
    )


def _aggregate_range(db: Session, user_id: str, start: datetime, end: datetime) -> UsageWindow:
    """임의 기간 합계."""
    requests_count, prompt_sum, completion_sum, total_sum, cost_sum = (
        db.query(
            func.count(UsageLog.id),
            func.coalesce(func.sum(UsageLog.prompt_tokens), 0),
            func.coalesce(func.sum(UsageLog.completion_tokens), 0),
            func.coalesce(func.sum(UsageLog.total_tokens), 0),
            func.coalesce(func.sum(UsageLog.cost), 0.0),
        )
        .filter(UsageLog.user_id == user_id, UsageLog.timestamp >= start, UsageLog.timestamp <= end)
        .one()
    )

    return UsageWindow(
        requests=int(requests_count),
        prompt_tokens=int(prompt_sum or 0),
        completion_tokens=int(completion_sum or 0),
        total_tokens=int(total_sum or 0),
        cost=float(cost_sum or 0.0),
    )


@router.get("/keys")
async def list_profile_keys(
    auth_result: Annotated[tuple[APIKey | None, bool, str | None], Depends(verify_jwt_or_api_key_or_master)],
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[str | None, Query(default=None, description="마스터 키 사용 시 조회할 user_id")],
) -> list[KeySummary]:
    """사용자의 API 키 메타 조회(평문 키 미노출)."""
    target_user_id = _resolve_target_user(auth_result, user)

    keys = (
        db.query(APIKey)
        .filter(APIKey.user_id == target_user_id)
        .order_by(APIKey.created_at.asc())
        .all()
    )

    return [
        KeySummary(
            id=str(key.id),
            key_name=str(key.key_name) if key.key_name else None,
            user_id=str(key.user_id) if key.user_id else None,
            created_at=key.created_at.isoformat() if key.created_at else None,
            last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
            expires_at=key.expires_at.isoformat() if key.expires_at else None,
            is_active=bool(key.is_active),
            metadata=dict(key.metadata_) if key.metadata_ else {},
        )
        for key in keys
    ]
