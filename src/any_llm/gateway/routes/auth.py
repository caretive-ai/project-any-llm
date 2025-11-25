import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from any_llm.gateway.auth import generate_api_key, hash_key, verify_jwt_or_api_key_or_master
from any_llm.gateway.auth.dependencies import get_config
from any_llm.gateway.auth.tokens import generate_refresh_token, hash_token, sign_access_token
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import APIKey, Budget, CaretUser, SessionToken, User, get_db

router = APIRouter(prefix="/v1/auth", tags=["auth"])


class SocialLoginRequest(BaseModel):
    """소셜 로그인 요청."""

    provider: str = Field(description="소셜 프로바이더 식별자 (예: google, kakao)")
    access_token: str = Field(description="소셜 프로바이더에서 발급받은 액세스 토큰 또는 id_token")
    email: str | None = Field(default=None, description="사용자 이메일(프로바이더에서 확보한 값, 없으면 None)")
    name: str | None = Field(default=None, description="사용자 이름/닉네임(없으면 None)")
    avatar_url: str | None = Field(default=None, description="사용자 아바타 이미지 URL(없으면 None)")
    device_type: str | None = Field(default=None, description="디바이스 유형 (mobile/desktop/web 등)")
    device_id: str | None = Field(default=None, description="디바이스 고유 식별자(선택)")
    os: str | None = Field(default=None, description="OS 정보(예: iOS 18, Android 15, macOS)")
    app_version: str | None = Field(default=None, description="앱/클라이언트 버전")
    user_agent: str | None = Field(default=None, description="브라우저/클라이언트 UA")
    ip: str | None = Field(default=None, description="클라이언트 IP(서버에서 주입 가능)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")


class RefreshRequest(BaseModel):
    """토큰 갱신 요청."""

    refresh_token: str


class LogoutRequest(BaseModel):
    """로그아웃 요청."""

    refresh_token: str


class BudgetInfo(BaseModel):
    """예산 요약."""

    budget_id: str
    max_budget: float | None
    budget_duration_sec: int | None


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

    profile: dict[str, Any]
    budget: BudgetInfo | None
    api_key_id: str | None


def _normalize_profile(request: SocialLoginRequest) -> dict[str, Any]:
    """소셜 토큰을 검증/정규화한다.

    실제 프로바이더 검증 로직은 이 함수에 통합해 교체한다.
    """
    return {
        "provider": request.provider,
        "email": request.email,
        "name": request.name,
        "avatar_url": request.avatar_url,
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
    api_key_id: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, str, datetime, datetime]:
    """access/refresh 토큰 발급 + 세션 저장."""
    jti = str(uuid.uuid4())
    access_token = sign_access_token(user_id=user_id, api_key_id=api_key_id, config=config, jti=jti)

    refresh_token = generate_refresh_token()
    refresh_hash = hash_token(refresh_token)
    now = datetime.now(UTC)
    refresh_exp = now + timedelta(days=config.refresh_token_exp_days)

    session = SessionToken(
        id=jti,
        user_id=user_id,
        api_key_id=api_key_id,
        refresh_token_hash=refresh_hash,
        refresh_expires_at=refresh_exp,
        created_at=now,
        last_used_at=now,
        metadata_=metadata or {},
    )
    db.add(session)
    return access_token, refresh_token, datetime.fromtimestamp(jwt_exp(access_token), tz=UTC), refresh_exp


def jwt_exp(token: str) -> int:
    """JWT exp 클레임을 추출."""
    payload = jwt.decode(token, options={"verify_signature": False})
    return int(payload["exp"])


@router.post("/login")
async def social_login(
    request: SocialLoginRequest,
    db: Annotated[Session, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> LoginResponse:
    """소셜 로그인 및 신규 가입."""
    profile = _normalize_profile(request)

    caret_user = db.query(CaretUser).filter(CaretUser.provider == profile["provider"]).first()

    is_new_user = caret_user is None
    budget: Budget
    user: User
    api_key: APIKey
    raw_api_key: str | None = None

    if is_new_user:
        budget = Budget(max_budget=1.0, budget_duration_sec=2_592_000)
        db.add(budget)
        db.flush()

        user = User(
            user_id=str(uuid.uuid4()),
            alias=profile.get("name"),
            budget_id=budget.budget_id,
            blocked=False,
            metadata_=request.metadata,
            budget_started_at=datetime.now(UTC),
            next_budget_reset_at=datetime.now(UTC) + timedelta(days=30),
        )
        db.add(user)
        db.flush()

        api_key, raw_api_key = _get_or_create_api_key(db, user.user_id, allow_create=True)

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
    else:
        user = db.query(User).filter(User.user_id == caret_user.user_id).first()
        if not user:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Linked user missing")

    access_token, refresh_token, access_exp, refresh_exp = _issue_tokens(
        config,
        db,
        user.user_id,
        api_key.id,
        metadata=profile.get("device"),
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
) -> TokenBundle:
    """Access/refresh 토큰 재발급."""
    refresh_hash = hash_token(request.refresh_token)
    session = (
        db.query(SessionToken)
        .filter(SessionToken.refresh_token_hash == refresh_hash)
        .first()
    )
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    now = datetime.now(UTC)
    if session.revoked_at:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token revoked")
    if session.refresh_expires_at and session.refresh_expires_at < now:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token expired")

    api_key = db.query(APIKey).filter(APIKey.id == session.api_key_id).first()
    if not api_key or not api_key.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key inactive")

    # Rotate refresh token
    session.revoked_at = now

    new_refresh = generate_refresh_token()
    new_refresh_hash = hash_token(new_refresh)
    refresh_exp = now + timedelta(days=config.refresh_token_exp_days)

    new_session = SessionToken(
        id=str(uuid.uuid4()),
        user_id=session.user_id,
        api_key_id=session.api_key_id,
        refresh_token_hash=new_refresh_hash,
        refresh_expires_at=refresh_exp,
        created_at=now,
        last_used_at=now,
        metadata_=session.metadata_ if session.metadata_ else {},
    )
    db.add(new_session)

    access_token = sign_access_token(
        user_id=session.user_id,
        api_key_id=session.api_key_id,
        config=config,
        jti=new_session.id,
    )

    db.commit()

    access_exp = datetime.fromtimestamp(jwt_exp(access_token), tz=UTC)
    return TokenBundle(
        access_token=access_token,
        access_token_expires_at=access_exp.isoformat(),
        refresh_token=new_refresh,
        refresh_token_expires_at=refresh_exp.isoformat(),
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: LogoutRequest,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """세션(리프레시 토큰) 폐기."""
    refresh_hash = hash_token(request.refresh_token)
    session = (
        db.query(SessionToken)
        .filter(SessionToken.refresh_token_hash == refresh_hash)
        .first()
    )
    if session:
        session.revoked_at = datetime.now(UTC)
        db.commit()


@router.get("/me")
async def me(
    auth_result: Annotated[tuple[APIKey | None, bool, str | None], Depends(verify_jwt_or_api_key_or_master)],
    db: Annotated[Session, Depends(get_db)],
) -> MeResponse:
    """내 정보 조회 (JWT/API 키)."""
    api_key, is_master, user_id = auth_result
    if is_master:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Master key is not allowed")

    resolved_user_id = user_id or (api_key.user_id if api_key else None)
    if not resolved_user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not resolved")

    user = db.query(User).filter(User.user_id == resolved_user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    caret_user = db.query(CaretUser).filter(CaretUser.user_id == resolved_user_id).first()
    budget = db.query(Budget).filter(Budget.budget_id == user.budget_id).first() if user.budget_id else None

    profile_payload: dict[str, Any] = {
        "user_id": user.user_id,
        "alias": user.alias,
        "provider": caret_user.provider if caret_user else None,
        "email": (caret_user.email if caret_user else None)
        or (dict(user.metadata_).get("email") if user.metadata_ else None),
        "name": caret_user.name if caret_user else None,
        "avatar_url": caret_user.avatar_url if caret_user else None,
        "blocked": bool(user.blocked),
        "spend": float(user.spend),
        "budget_id": user.budget_id,
        "user_metadata": dict(user.metadata_) if user.metadata_ else {},
        "provider_metadata": dict(caret_user.metadata_) if caret_user and caret_user.metadata_ else {},
    }

    return MeResponse(
        profile=profile_payload,
        budget=(
            BudgetInfo(
                budget_id=budget.budget_id,
                max_budget=budget.max_budget,
                budget_duration_sec=budget.budget_duration_sec,
            )
            if budget
            else None
        ),
        api_key_id=api_key.id if api_key else None,
    )
