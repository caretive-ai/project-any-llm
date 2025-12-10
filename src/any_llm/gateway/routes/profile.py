from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from any_llm.gateway.auth import verify_jwt_or_api_key_or_master
from any_llm.gateway.db import (
    APIKey,
    CaretUser,
    CreditBalance,
    SessionToken,
    UsageLog,
    User,
    get_db,
)
from sqlalchemy import text
from any_llm.gateway.routes.utils import resolve_target_user
from any_llm.gateway.log_config import logger

router = APIRouter(prefix="/v1/profile", tags=["profile"])


class ProfileInfo(BaseModel):
    """사용자/소셜 통합 프로필."""

    user_id: str
    provider: str | None
    role: str | None
    name: str | None
    email: str | None
    avatar_url: str | None
    refresh_token: str | None
    blocked: bool
    spend: float
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
    usage: dict[str, UsageWindow]


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


class LogItem(BaseModel):
    """사용 로그 아이템."""

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
    api_key_id: str | None
    user_id: str | None


class LogsResponse(BaseModel):
    """사용 로그 응답."""

    total: int
    items: list[LogItem]

class Balance(BaseModel):
    """사용자 잔여 크레딧."""
    userId: str
    balance: float


class BalanceResponse(BaseModel):
    """크레딧 잔액 응답."""
    success: bool
    data: Balance

@router.get("/balance", response_model=BalanceResponse)
async def get_balance(
    auth_result: Annotated[tuple[APIKey | None, bool, str | None, SessionToken | None], Depends(verify_jwt_or_api_key_or_master)],
    db: Annotated[Session, Depends(get_db)],
) -> BalanceResponse:
    """현재 사용자의 크레딧 잔액을 반환."""
    target_user_id = resolve_target_user(
        auth_result,
        None,
        missing_master_detail="When using master key, user resolution failed",
    )

    # row = (
    #     db.query(func.coalesce(func.sum(func.cast(User.spend, Float)), 0).label("spend"))
    #     .filter(User.user_id == target_user_id)
    #     .one_or_none()
    # )

    # billing_credit_balances.total_credits를 우선, 없으면 spend 기준 0
    available_credits = db.query(func.coalesce(func.sum(CreditBalance.amount), 0.0)).filter(
        CreditBalance.user_id == target_user_id,
        (CreditBalance.expires_at.is_(None)) | (CreditBalance.expires_at > func.now()),
    ).scalar() or 0.0
    logger.info("User %s credit balance: %s", target_user_id, available_credits)

    return BalanceResponse(
        success=True,
        data=Balance(
            userId=target_user_id,
            balance=float(available_credits) * 100 * 1000, # micro dollors,
        ),
    )


def _now_naive() -> datetime:
    """UTC now without tzinfo (DB 저장 방식과 동일)."""
    return datetime.now(UTC).replace(tzinfo=None)


def _ensure_naive(dt: datetime) -> datetime:
    """Ensure datetime is naive UTC."""
    if dt.tzinfo:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


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
    limit = limit or 20  # Default to 20 when 0/None provided
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
    auth_result: Annotated[tuple[APIKey | None, bool, str | None, SessionToken | None], Depends(verify_jwt_or_api_key_or_master)],
    db: Annotated[Session, Depends(get_db)],
    user: str | None = Query(None, description="마스터 키 사용 시 조회할 user_id"),
) -> ProfileResponse:
    """프로필 + 예산 + 사용량 집계 반환."""
    target_user_id = resolve_target_user(
        auth_result,
        user,
        missing_master_detail="When using master key, 'user' query parameter is required",
    )
    user_obj = db.query(User).filter(User.user_id == target_user_id).first()
    if not user_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User '{target_user_id}' not found")

    session_token = (
        db.query(SessionToken)
        .filter(SessionToken.user_id == target_user_id, SessionToken.revoked_at.is_(None))
        .order_by(SessionToken.created_at.desc())
        .first()
    )

    caret = db.query(CaretUser).filter(CaretUser.user_id == target_user_id).first()

    now = _now_naive()
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_of_week = start_of_day - timedelta(days=start_of_day.weekday())
    start_of_month = start_of_day.replace(day=1)
    usage_windows = {
        "today": _aggregate_range(db, target_user_id, start_of_day, now),
        "this_week": _aggregate_range(db, target_user_id, start_of_week, now),
        "this_month": _aggregate_range(db, target_user_id, start_of_month, now),
    }

    profile = ProfileInfo(
        user_id=user_obj.user_id,
        provider=caret.provider if caret else None,
        role=caret.role if caret else None,
        spend=float(user_obj.spend),
        name=caret.name if caret else user_obj.alias,
        email=caret.email if caret else None,
        avatar_url=caret.avatar_url if caret else None,
        refresh_token=session_token.refresh_token_plain if session_token else None,
        blocked=bool(user_obj.blocked),
        metadata=dict(user_obj.metadata_) if user_obj.metadata_ else {},
        caret_metadata=dict(caret.metadata_) if caret and caret.metadata_ else {},
        last_login_at=caret.last_login_at.isoformat() if caret and caret.last_login_at else None,
    )

    return ProfileResponse(
        profile=profile,
        usage=usage_windows,
    )


@router.get("/usage")
async def get_profile_usage(
    auth_result: Annotated[tuple[APIKey | None, bool, str | None, SessionToken | None], Depends(verify_jwt_or_api_key_or_master)],
    db: Annotated[Session, Depends(get_db)],
    user: str | None = Query(None, description="마스터 키 사용 시 조회할 user_id"),
    start: datetime | None = Query(None, description="집계 시작 시각(ISO). 기본: now-30d"),
    end: datetime | None = Query(None, description="집계 종료 시각(ISO). 기본: now"),
    group_by: Literal["day", "week", "total"] = Query("day", description="집계 단위"),
) -> UsageBucketsResponse:
    """사용량 집계(기간별)."""
    target_user_id = resolve_target_user(
        auth_result,
        user,
        missing_master_detail="When using master key, 'user' query parameter is required",
    )

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
    auth_result: Annotated[tuple[APIKey | None, bool, str | None, SessionToken | None], Depends(verify_jwt_or_api_key_or_master)],
    db: Annotated[Session, Depends(get_db)],
    user: str | None = Query(None, description="마스터 키 사용 시 조회할 user_id"),
) -> list[KeySummary]:
    """사용자의 API 키 메타 조회(평문 키 미노출)."""
    target_user_id = resolve_target_user(
        auth_result,
        user,
        missing_master_detail="When using master key, 'user' query parameter is required",
    )

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


class ProfileLog(BaseModel):
    aiInferenceProviderName: str
    aiModelName: str
    aiModelTypeName: str
    completionTokens: int
    costUsd: float
    createdAt: str
    creditsUsed: float
    generationId: str
    id: str
    metadata: dict[str, Any]
    organizationId: str
    promptTokens: int
    totalTokens: int
    userId: str

class ProfileLogResponse(BaseModel):
    success: bool
    data: dict[str, Any] = {
        "items": list[ProfileLog]
    }


class PaymentRecord(BaseModel):
    paidAt: str
    creatorId: str
    amountCents: int
    credits: int

class PaymentResponse(BaseModel):
    success: bool
    data: dict[str, Any] = {
        "paymentTransactions": list[PaymentRecord]
    }

@router.get("/logs")
async def list_profile_logs(
    auth_result: Annotated[tuple[APIKey | None, bool, str | None, SessionToken | None], Depends(verify_jwt_or_api_key_or_master)],
    db: Annotated[Session, Depends(get_db)],
) -> ProfileLogResponse:
    """현재 날짜 기준 최근 7일 로그(최대 100건, 필터 없음)."""
    target_user_id = resolve_target_user(
        auth_result,
        None,
        missing_master_detail="When using master key, 'user' query parameter is required",
    )

    end_dt = datetime.now(UTC).replace(tzinfo=None)
    start_dt = end_dt - timedelta(days=7)

    logs = (
        db.query(UsageLog)
        .filter(UsageLog.user_id == target_user_id, UsageLog.timestamp >= start_dt)
        .order_by(UsageLog.timestamp.desc())
        .limit(100)
        .all()
    )

    return ProfileLogResponse(
        success=True,
        data={
            "items": [
              ProfileLog(
                  aiInferenceProviderName=log.provider or "",
                  aiModelName=log.model or "",
                  aiModelTypeName="",
                  completionTokens=log.completion_tokens or 0,
                  costUsd=log.cost or 0.0,
                  createdAt=log.timestamp.isoformat() if log.timestamp else "",
                  creditsUsed=(log.cost or 0.0) * 1000000, # micro
                  generationId=log.id or "",
                  id=log.id or "",
                  metadata={},
                  organizationId="",
                  promptTokens=log.prompt_tokens or 0,
                  totalTokens=log.total_tokens or 0,
                  userId=log.user_id or "",
              )
              for log in logs
          ]
        }
      )


@router.get("/payments")
async def list_profile_payments(
    auth_result: Annotated[tuple[APIKey | None, bool, str | None, SessionToken | None], Depends(verify_jwt_or_api_key_or_master)],
    db: Annotated[Session, Depends(get_db)],
) -> PaymentResponse:
    """현재 사용자 결제 이력(청구 인보이스 기반) 반환."""
    target_user_id = resolve_target_user(
        auth_result,
        None,
        missing_master_detail="When using master key, 'user' query parameter is required",
    )

    rows = db.execute(
        text(
            """
            SELECT user_id, total AS amount_cents, credits, created_at
            FROM billing_invoices
            WHERE user_id = :uid
            ORDER BY created_at DESC
            """
        ),
        {"uid": target_user_id},
    ).fetchall()

    payments: list[PaymentRecord] = []
    for row in rows:
        # Support tuple/Row/_mapping depending on SQLAlchemy version/driver
        row_map = row._mapping if hasattr(row, "_mapping") else row
        amount_raw = row_map["amount_cents"] if "amount_cents" in row_map else None
        credits_raw = row_map["credits"] if "credits" in row_map else None
        created_raw = row_map["created_at"] if "created_at" in row_map else None

        amount = int(amount_raw) if amount_raw is not None else 0
        credits = int(credits_raw) if credits_raw is not None else 0
        created_at = created_raw.isoformat() if created_raw else ""

        payments.append(
            PaymentRecord(
                paidAt=created_at,
                creatorId=row_map["user_id"] if "user_id" in row_map and row_map["user_id"] else "",
                amountCents=amount,
                credits=credits,
            )
        )

    return PaymentResponse(
      success=True,
      data={
        "paymentTransactions": payments
      }
    )
