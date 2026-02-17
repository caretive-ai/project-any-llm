import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import jwt
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from any_llm.gateway.auth import generate_api_key, hash_key, verify_jwt_or_api_key_or_master
from any_llm.gateway.auth.dependencies import get_config
from any_llm.gateway.auth.tokens import generate_refresh_token, hash_token, sign_access_token
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import (
    APIKey,
    BillingPlan,
    BillingSubscription,
    CaretUser,
    CreditBalance,
    CreditTopup,
    SessionToken,
    User,
    get_db,
)
from any_llm.gateway.log_config import logger

router = APIRouter(prefix="/v1/auth", tags=["auth"])


def _mask_token(token: str | None, *, keep: int = 4) -> str:
    """Return a masked token representation for safe logging."""
    if not token:
        return "<empty>"
    token = token.strip()
    if len(token) <= keep * 2:
        return f"{token[:keep]}...len={len(token)}"
    return f"{token[:keep]}...{token[-keep:]}(len={len(token)})"


class SocialLoginRequest(BaseModel):
    """소셜 로그인 요청."""

    provider: str = Field(description="소셜 프로바이더 식별자 (예: google, kakao)")
    email: str | None = Field(default=None, description="사용자 이메일(프로바이더에서 확보한 값, 없으면 None)")
    name: str | None = Field(default=None, description="사용자 이름/닉네임(없으면 None)")
    avatar_url: str | None = Field(default=None, description="사용자 아바타 이미지 URL(없으면 None)")
    device_type: str | None = Field(default=None, description="디바이스 유형 (mobile/desktop/web 등)")
    device_id: str | None = Field(default=None, description="디바이스 고유 식별자(선택)")
    os: str | None = Field(default=None, description="OS 정보(예: iOS 18, Android 15, macOS)")
    app_version: str | None = Field(default=None, description="앱/클라이언트 버전")
    user_agent: str | None = Field(default=None, description="브라우저/클라이언트 UA")
    ip: str | None = Field(default=None, description="클라이언트 IP(서버에서 주입 가능)")
    provider_token: str | None = Field(default=None, description="소셜 프로바이더 토큰")
    metadata: dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")


class RefreshRequest(BaseModel):
    """토큰 갱신 요청."""

    refreshToken: str
    source: str | None = None


class TokenRequest(BaseModel):
    """Provider code로 토큰 교환 요청."""

    code: str
    provider: str | None = None
    client_type: str | None = None


class TokenResponse(BaseModel):
    """토큰 교환 응답."""
    success: bool
    data: dict[str, Any] = {
      "accessToken": str,
      "refreshToken": str,
      "expiresAt": str | None,
      "userInfo": dict[str, Any] | None,
      "tokenType": str | None,
    }

class LogoutRequest(BaseModel):
    """로그아웃 요청."""

    refresh_token: str



class TokenBundle(BaseModel):
    """토큰 응답."""

    access_token: str
    access_token_expires_at: str
    refresh_token: str
    refresh_token_expires_at: str


class LoginResponse(BaseModel):
    """로그인/가입 응답."""

    tokens: TokenBundle


class MeResponse(BaseModel):
    """내 정보 응답."""

    success: bool
    data: dict[str, Any] = {  
      "id": str,
      "email": str,
      "displayName": str,
      "photoUrl": str,
      "createdAt": str,
      "updatedAt": str,
      "organizations": list[dict[str, Any]]
    }


class AuthorizeResponse(BaseModel):
    """인가 요청 응답."""

    redirect_url: str


def _normalize_refresh_token(raw: str) -> str:
    """Strip bearer prefix/spaces to accept tokens stored with or without 'Bearer '."""
    token = raw.strip()
    if token.lower().startswith("bearer "):
        return token[7:].strip()
    return token


def _normalize_profile(request: SocialLoginRequest) -> dict[str, Any]:
    """소셜 토큰을 검증/정규화한다.

    실제 프로바이더 검증 로직은 이 함수에 통합해 교체한다.
    """
    return {
        "provider": request.provider,
        "email": request.email,
        "name": request.name,
        "avatar_url": request.avatar_url,
        "provider_token": request.provider_token,
        "metadata": request.metadata or {},
        "device": {
            "device_type": request.device_type or "web",
            "device_id": request.device_id or "",
            "os": request.os or "",
            "app_version": request.app_version or "",
            "user_agent": request.user_agent or "",
            "ip": request.ip or "",
        },
    }


def _get_or_create_api_key(db: Session, user_id: str, allow_create: bool) -> tuple[APIKey, str | None]:
    """사용자용 API 키를 가져오거나 생성."""
    api_key = (
        db.query(APIKey)
        .filter(APIKey.user_id == user_id, APIKey.is_active.is_(True))
        .order_by(APIKey.created_at.asc())
        .first()
    )
    if api_key:
        return api_key, None
    if not allow_create:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not found for existing user",
        )

    raw_key = generate_api_key()
    api_key = APIKey(
        id=str(uuid.uuid4()),
        key_hash=hash_key(raw_key),
        key_name="default",
        user_id=user_id,
    )
    db.add(api_key)
    db.flush()
    return api_key, raw_key


def _issue_tokens(
    config: GatewayConfig,
    db: Session,
    user_id: str,
    metadata: dict[str, Any] | None = None,
    provider_token: str | None = None,
) -> tuple[str, str, datetime, datetime]:
    """access/refresh 토큰 발급 + 세션 저장."""
    jti = str(uuid.uuid4())
    access_token = sign_access_token(user_id=user_id, config=config, jti=jti)

    refresh_token = generate_refresh_token()
    refresh_hash = hash_token(refresh_token)
    now = datetime.now(UTC)
    refresh_exp = now + timedelta(days=config.refresh_token_exp_days)

    merged_metadata = metadata or {}

    session = SessionToken(
        id=jti,
        user_id=user_id,
        refresh_token_hash=refresh_hash,
        refresh_token_plain=refresh_token,
        access_token_plain=access_token,
        provider_token=provider_token,
        refresh_expires_at=refresh_exp,
        created_at=now,
        last_used_at=now,
        metadata_=merged_metadata,
    )
    db.add(session)
    return access_token, refresh_token, datetime.fromtimestamp(jwt_exp(access_token), tz=UTC), refresh_exp


def jwt_exp(token: str) -> int:
    """JWT exp 클레임을 추출."""
    payload = jwt.decode(token, options={"verify_signature": False})
    return int(payload["exp"])


def _ensure_free_plan_and_balance(db: Session, user_id: str, now: datetime) -> None:
    """Seed FREE plan subscription and signup bonus for new users.

    Matching careti.ai FREE plan:
    - Signup: 20 credits instantly (BONUS pool, no expiry)
    - Monthly: 10 credits auto-top-up if balance < 10 (handled by renewal cron)
    """
    plan = (
        db.query(BillingPlan)
        .filter(BillingPlan.name == "FREE", BillingPlan.active.is_(True))
        .first()
    )
    if not plan:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="FREE plan not configured")

    renew_at = now + timedelta(days=int(plan.renew_interval_days))
    subscription = BillingSubscription(
        user_id=user_id,
        plan_id=plan.id,
        status="ACTIVE",
        start_at=now,
        renew_at=renew_at,
    )
    db.add(subscription)

    # Signup bonus: 20 credits (monthly_bonus_credits), no expiry
    bonus_balance = CreditBalance(
        user_id=user_id,
        pool_type="BONUS",
        source_id=plan.id,
        amount=float(plan.monthly_bonus_credits),
        expires_at=None,
        priority=1,
    )
    db.add(bonus_balance)

    bonus_topup = CreditTopup(
        user_id=user_id,
        pool_type="BONUS",
        amount=float(plan.monthly_bonus_credits),
        amount_usd=0.0,
        credits_per_usd=float(plan.credits_per_usd),
        expires_at=None,
        source="FIRST_JOIN",
        metadata_={"seeded": True},
    )
    db.add(bonus_topup)

@router.get("/authorize", response_model=AuthorizeResponse)
async def authorize(
    callback_url: Annotated[str, Query(..., description="로그인 후 돌아올 콜백 URL (예: vscode://caretive.caret/auth)")],
    client_type: Annotated[str, Query(description="클라이언트 유형")] = "extension",
    redirect_uri: Annotated[str | None, Query(description="명시적 리디렉션 URI (없으면 callback_url 사용)")] = None,
    config: Annotated[GatewayConfig | None, Depends(get_config)] = None,
) -> AuthorizeResponse:
    """인가 리디렉트 URL을 생성해 반환."""
    if not callback_url:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="callback_url is required")

    base = config.auth_base_url if config else "http://localhost:4001"
    params = {
        "client_type": client_type or "extension",
        "callback_url": callback_url,
    }
    redirect_url = f"{base}/login?{urlencode(params)}"
    return AuthorizeResponse(redirect_url=redirect_url)


@router.post("/token", response_model=TokenResponse)
async def exchange_token(
    request: TokenRequest,
    db: Annotated[Session, Depends(get_db)],
) -> TokenResponse:
    """provider code로 저장된 세션을 조회해 access/refresh 토큰을 반환."""
    if not request.code:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="code is required")

    session: Any = (
        db.query(SessionToken, CaretUser)
        .join(CaretUser, CaretUser.user_id == SessionToken.user_id, isouter=True)
        .filter(SessionToken.provider_token == request.code)
        .order_by(SessionToken.created_at.desc())
        .first()
    )
    now = datetime.now(UTC)
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or expired code")

    token_row: SessionToken = session.SessionToken
    caret_row: CaretUser | None = session.CaretUser

    if token_row.refresh_expires_at < now or token_row.revoked_at:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or expired code")

    if not token_row.refresh_token_plain or not token_row.access_token_plain:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="session missing tokens")
    user_info = {
        "id": caret_row.id if caret_row else "",
        "email": caret_row.email if caret_row else "",
        "name": caret_row.name if caret_row else "",
        "subject": caret_row.provider if caret_row else "",
        "photoUrl": caret_row.avatar_url if caret_row else "",
        "accounts": [],
    }

    try:
        access_exp = datetime.fromtimestamp(jwt_exp(token_row.access_token_plain), tz=UTC)
        expires_at = access_exp.isoformat()
    except Exception:
        expires_at = None

    return TokenResponse(
        success=True,
        data={
            "accessToken": token_row.access_token_plain,
            "refreshToken": token_row.refresh_token_plain,
            "tokenType": "Bearer",
            "expiresAt": expires_at,
            "userInfo": user_info,
        },
    )


@router.post("/login")
async def social_login(
    request: SocialLoginRequest,
    db: Annotated[Session, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> LoginResponse:
    """소셜 로그인 및 신규 가입."""
    profile = _normalize_profile(request)
    logger.info(f"Social login: {profile}")

    caret_user = (
        db.query(CaretUser)
        .filter(
            CaretUser.provider == profile["provider"],
            CaretUser.email == profile["email"],
        )
        .first()
    )

    is_new_user = caret_user is None
    user: User | None
    api_key: APIKey
    raw_api_key: str | None = None

    if is_new_user:
        now = datetime.now(UTC)
        user = User(
            user_id=str(uuid.uuid4()),
            alias=profile.get("name"),
            blocked=False,
            metadata_=request.metadata,
        )
        db.add(user)
        db.flush()

        # api_key, raw_api_key = _get_or_create_api_key(db, user.user_id, allow_create=True)

        caret_user = CaretUser(
            user_id=user.user_id,
            provider=profile["provider"],
            email=profile.get("email"),
            role="user",
            name=profile.get("name"),
            avatar_url=profile.get("avatar_url"),
            metadata_=profile.get("metadata") or {},
            last_login_at=datetime.now(UTC),
        )
        db.add(caret_user)

        _ensure_free_plan_and_balance(db, user.user_id, now)
    else:
        user = db.query(User).filter(User.user_id == caret_user.user_id).first()
        if not user:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Linked user missing")

    access_token, refresh_token, access_exp, refresh_exp = _issue_tokens(
        config,
        db,
        user.user_id,
        metadata=profile.get("device"),
        provider_token=profile.get("provider_token"),
    )
    db.commit()

    return LoginResponse(
        tokens=TokenBundle(
            access_token=access_token,
            access_token_expires_at=access_exp.isoformat(),
            refresh_token=refresh_token,
            refresh_token_expires_at=refresh_exp.isoformat(),
        ),
    )


@router.post("/refresh")
async def refresh_token(
    request: RefreshRequest,
    db: Annotated[Session, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> TokenResponse:
    """Access/refresh 토큰 재발급."""
    logger.info("refresh_token request received raw=%s", _mask_token(request.refreshToken))
    normalized_refresh = _normalize_refresh_token(request.refreshToken)
    logger.info("refresh_token normalized=%s", _mask_token(normalized_refresh))
    refresh_hash = hash_token(normalized_refresh)
    logger.info("refresh_token hash=%s", refresh_hash)
    session = (
        db.query(SessionToken, CaretUser)
        .join(CaretUser, CaretUser.user_id == SessionToken.user_id, isouter=True)
        .filter(SessionToken.refresh_token_hash == refresh_hash)
        .first()
    )
    if not session:
        logger.warning("refresh_token session lookup failed hash=%s", refresh_hash)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    session_row: SessionToken = session.SessionToken
    caret_row: CaretUser | None = session.CaretUser

    now = datetime.now(UTC)
    if session_row.revoked_at and request.source != "home":
        logger.info(
            "refresh_token denied: revoked session_id=%s source=%s",
            session_row.id,
            request.source,
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token revoked")
    if session_row.refresh_expires_at and session_row.refresh_expires_at < now:
        logger.info(
            "refresh_token denied: expired session_id=%s refresh_expires_at=%s",
            session_row.id,
            session_row.refresh_expires_at.isoformat(),
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token expired")

    # Rotate refresh token
    session_row.revoked_at = now

    new_refresh = generate_refresh_token()
    new_refresh_hash = hash_token(new_refresh)
    refresh_exp = now + timedelta(days=config.refresh_token_exp_days)

    # Issue new access token first so we can store the plain value on the session row
    new_session_id = str(uuid.uuid4())
    access_token = sign_access_token(
        user_id=session_row.user_id,
        config=config,
        jti=new_session_id,
    )

    new_session = SessionToken(
        id=new_session_id,
        user_id=session_row.user_id,
        refresh_token_hash=new_refresh_hash,
        refresh_token_plain=new_refresh,
        access_token_plain=access_token,
        refresh_expires_at=refresh_exp,
        created_at=now,
        last_used_at=now,
        metadata_=session_row.metadata_ if session_row.metadata_ else {},
    )
    db.add(new_session)
    db.commit()
    logger.info(
        "refresh_token rotated old_session_id=%s new_session_id=%s user_id=%s new_refresh=%s refresh_expires_at=%s",
        session_row.id,
        new_session_id,
        session_row.user_id,
        _mask_token(new_refresh),
        refresh_exp.isoformat(),
    )

    access_exp = datetime.fromtimestamp(jwt_exp(access_token), tz=UTC)
    user_info = {
        "id": caret_row.id if caret_row else "",
        "email": caret_row.email if caret_row else "",
        "name": caret_row.name if caret_row else "",
        "subject": caret_row.provider if caret_row else "",
        "photoUrl": caret_row.avatar_url if caret_row else "",
        "accounts": [],
    }

    return TokenResponse(
        success=True,
        data={
            "accessToken": access_token,
            "refreshToken": new_refresh,
            "tokenType": "Bearer",
            "expiresAt": access_exp.isoformat(),
            "userInfo": user_info,
        },
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: LogoutRequest,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """세션(리프레시 토큰) 폐기."""
    logger.debug("logout request received raw=%s", _mask_token(request.refresh_token))
    normalized_refresh = _normalize_refresh_token(request.refresh_token)
    refresh_hash = hash_token(normalized_refresh)
    logger.debug(
        "logout normalized=%s hash=%s",
        _mask_token(normalized_refresh),
        refresh_hash,
    )
    session = (
        db.query(SessionToken)
        .filter(SessionToken.refresh_token_hash == refresh_hash)
        .first()
    )
    if session:
        logger.debug("logout session revoked session_id=%s user_id=%s", session.id, session.user_id)
        session.revoked_at = datetime.now(UTC)
        db.commit()
    else:
        logger.warning("logout session not found hash=%s", refresh_hash)


@router.get("/me")
async def me(
    auth_result: Annotated[tuple[APIKey | None, bool, str | None, SessionToken | None], Depends(verify_jwt_or_api_key_or_master)],
    db: Annotated[Session, Depends(get_db)],
) -> MeResponse:
    """내 정보 조회 (JWT/API 키)."""
    api_key, is_master, user_id, _ = auth_result
    logger.warning(f"User ID: {user_id}")

    if is_master:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Master key is not allowed")

    resolved_user_id = user_id or (api_key.user_id if api_key else None)
    if not resolved_user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not resolved")

    user = db.query(User).filter(User.user_id == resolved_user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    caret_user = db.query(CaretUser).filter(CaretUser.user_id == resolved_user_id).first()
    if not caret_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CaretUser profile not found")

    return MeResponse(
            success=True,
            data={
              "id":caret_user.id,
              "email":caret_user.email,
              "displayName":caret_user.name,
              "photoUrl":caret_user.avatar_url,
              "createdAt":caret_user.created_at.isoformat(),
              "updatedAt":caret_user.updated_at.isoformat(),
              "organizations": [],
            }
        )
