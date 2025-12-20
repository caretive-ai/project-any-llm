from datetime import UTC, datetime
from typing import Any, Tuple

from fastapi import HTTPException, status
from sqlalchemy import func, update
from sqlalchemy.orm import Session

from any_llm.gateway.db import (
    APIKey,
    BillingPlan,
    BillingSubscription,
    CreditBalance,
    CreditCharge,
    ModelPricing,
    SessionToken,
)
from any_llm.gateway.log_config import logger

CREDITS_PER_USD_DEFAULT = 10.0


def _get_cached_prompt_tokens(usage: Any) -> int | None:
    """Extract cached prompt tokens from a provider usage object.

    Supports:
    - any-llm extensions: usage.cached_tokens (or legacy usage.cache_tokens)
    - OpenAI SDK: usage.prompt_tokens_details.cached_tokens
    """
    if not usage:
        return None

    cached_tokens = getattr(usage, "cached_tokens", None)
    if cached_tokens is not None:
        try:
            return int(cached_tokens or 0)
        except (TypeError, ValueError):
            return None

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_details:
        if isinstance(prompt_details, dict):
            cached_tokens = prompt_details.get("cached_tokens")
        else:
            cached_tokens = getattr(prompt_details, "cached_tokens", None)
        if cached_tokens is not None:
            try:
                return int(cached_tokens or 0)
            except (TypeError, ValueError):
                return None

    return None


def _estimate_cost_usd(
    pricing: ModelPricing,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Convert prompt/completion counts to USD using pricing."""
    cached_tokens = min(max(cached_tokens, 0), prompt_tokens)
    cached_price = pricing.cached_price_per_million
    if cached_price is None:
        cached_price = pricing.input_price_per_million

    non_cached_prompt_tokens = prompt_tokens - cached_tokens
    return (
        (non_cached_prompt_tokens / 1_000_000) * pricing.input_price_per_million
        + (cached_tokens / 1_000_000) * cached_price
        + (completion_tokens / 1_000_000) * pricing.output_price_per_million
    )

def _estimate_cost_image_usd(
    pricing: ModelPricing,
    prompt_tokens: int,
    completion_tokens: int,
    thought_tokens: int = 0,
) -> float:
    """Convert prompt/completion counts to USD using pricing."""
    thought_price = pricing.cached_price_per_million
    if thought_price is None:
        thought_price = pricing.input_price_per_million

    return (
        (prompt_tokens / 1_000_000) * pricing.input_price_per_million
        + (completion_tokens / 1_000_000) * pricing.output_price_per_million
        + (thought_tokens / 1_000_000) *thought_price
    )    

def resolve_target_user(
    auth_result: Tuple[APIKey | None, bool, str | None, SessionToken | None],
    explicit_user: str | None,
    *,
    missing_master_detail: str = "When using master key, user is required",
) -> str:
    """Resolve a target user_id from auth context and optional explicit value."""
    api_key, is_master, resolved_user_id, _ = auth_result

    if is_master:
        if not explicit_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=missing_master_detail,
            )
        return explicit_user

    target_user_id = resolved_user_id or explicit_user or (api_key.user_id if api_key else None)
    if not target_user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not resolved")
    return target_user_id


def validate_user_credit(db: Session, user_id: str) -> None:
    """Ensure user has available (non-expired) credits; expire stale pools on the fly."""
    # Expire stale pools in one UPDATE to avoid loading all rows.
    db.execute(
        update(CreditBalance)
        .where(CreditBalance.user_id == user_id)
        .where(CreditBalance.expires_at.isnot(None))
        .where(CreditBalance.expires_at <= func.now())
        .where(CreditBalance.amount != 0)
        .values(amount=0),
        execution_options={"synchronize_session": False},
    )
    db.commit()

    available = db.query(func.coalesce(func.sum(CreditBalance.amount), 0)).filter(
        CreditBalance.user_id == user_id,
        (CreditBalance.expires_at.is_(None)) | (CreditBalance.expires_at > func.now()),
    ).scalar() or 0

    if available <= 0:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient credits")


def get_user_credits_per_usd(db: Session, user_id: str) -> float:
    """Fetch credits_per_usd from the user's active billing plan, fallback to default."""
    result = (
        db.query(BillingPlan.credits_per_usd)
        .join(BillingSubscription, BillingSubscription.plan_id == BillingPlan.id)
        .filter(
            BillingSubscription.user_id == user_id,
            BillingSubscription.status == "ACTIVE",
        )
        .scalar()
    )
    return float(result) if result is not None else CREDITS_PER_USD_DEFAULT


def charge_user_credits(
    db: Session,
    user_id: str,
    *,
    cost_usd: float,
    credits_per_usd: float = CREDITS_PER_USD_DEFAULT,
    model_key: str | None = None,
    unit_price_usd: float | None = None,
    usage_id: str | None = None,
    currency: str = "USD",
) -> float:
    """Deduct credits according to priority/expires_at and record credit_charges."""
    if cost_usd < 0:
        msg = "cost_usd must be non-negative"
        raise ValueError(msg)
    needed_credits = cost_usd * credits_per_usd
    if needed_credits <= 0:
        return 0.0

    # Lock rows to avoid race; ordered by priority ASC, expires_at ASC NULLS LAST.
    balances = (
        db.query(CreditBalance)
        .filter(
            CreditBalance.user_id == user_id,
            (CreditBalance.expires_at.is_(None)) | (CreditBalance.expires_at > func.now()),
            CreditBalance.amount > 0,
        )
        .order_by(CreditBalance.priority.asc(), CreditBalance.expires_at.is_(None), CreditBalance.expires_at.asc())
        .with_for_update(skip_locked=True)
        .all()
    )

    remaining = needed_credits
    charges: list[CreditCharge] = []

    for bal in balances:
        if remaining <= 0:
            break
        take = min(bal.amount, remaining)
        bal.amount -= take
        remaining -= take
        charges.append(
            CreditCharge(
                user_id=user_id,
                pool_id=bal.id,
                debited_amount=take,
                cost_usd=cost_usd,
                credits_per_usd=credits_per_usd,
                model_key=model_key,
                unit_price_usd=unit_price_usd,
                usage_id=usage_id,
                currency=currency,
            )
        )

    charged_credits = needed_credits - remaining
    if remaining > 0:
        logger.warning(
            "Insufficient credits to fully charge usage",
            extra={
                "user_id": user_id,
                "needed_credits": needed_credits,
                "charged_credits": charged_credits,
                "shortage": remaining,
                "model_key": model_key,
            },
        )
    for charge in charges:
        db.add(charge)
    db.commit()
    return charged_credits



def charge_usage_cost(
    db: Session,
    user_id: str,
    *,
    usage: Any,
    model_key: str | None,
    usage_id: str | None = None,
    currency: str = "USD",
) -> float:
    """usage(토큰)와 ModelPricing을 가져와 비용을 계산하고 크레딧을 차감한다."""
    if not usage or not model_key:
        return 0.0

    pricing = db.query(ModelPricing).filter(ModelPricing.model_key == model_key).first()
    if not pricing:
        logger.warning(f"No pricing configured for model '{model_key}'. Skipping credit charge.")
        return 0.0

    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    thought = getattr(usage, "thought_tokens", 0) or 0
    if thought > 0:
        cost_usd = _estimate_cost_image_usd(pricing, prompt, completion, thought)
    else:
        cached = _get_cached_prompt_tokens(usage) or 0
        cost_usd = _estimate_cost_usd(pricing, prompt, completion, cached)

    if cost_usd <= 0:
        return 0.0

    try:
        credits_per_usd = get_user_credits_per_usd(db, user_id)
        charge_user_credits(
            db,
            user_id=user_id,
            cost_usd=cost_usd,
            credits_per_usd=credits_per_usd,
            model_key=model_key,
            unit_price_usd=None,
            usage_id=usage_id,
            currency=currency,
        )
    except Exception as e:
        logger.warning("Credit charge failed", extra={"user_id": user_id, "model_key": model_key, "error": str(e)})

    return cost_usd
