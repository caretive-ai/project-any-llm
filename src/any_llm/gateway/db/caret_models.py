"""Caret-specific billing/credit models kept separate from core models."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from any_llm.gateway.db.models import Base


class SessionToken(Base):
    """Refresh/session token state for JWT-based access tokens."""

    __tablename__ = "session_tokens"

    id: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("caret_users.user_id", ondelete="CASCADE"), index=True)
    refresh_token_hash: Mapped[str] = mapped_column(unique=True, index=True)
    refresh_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    refresh_token_plain: Mapped[str | None] = mapped_column(String)
    provider_token: Mapped[str | None] = mapped_column(String, nullable=True)
    access_token_plain: Mapped[str | None] = mapped_column(String, nullable=True)
    caret_user = relationship("CaretUser", back_populates="session_tokens")

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "refresh_expires_at": self.refresh_expires_at.isoformat() if self.refresh_expires_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "metadata": self.metadata_,
            "refresh_token_plain": self.refresh_token_plain,
            "provider_token": self.provider_token,
            "access_token_plain": self.access_token_plain,
        }


class CaretUser(Base):
    """Caret user mapping between social provider and gateway user."""

    __tablename__ = "caret_users"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), unique=True, index=True)
    provider: Mapped[str] = mapped_column(index=True)
    role: Mapped[str] = mapped_column()
    email: Mapped[str | None] = mapped_column()
    name: Mapped[str | None] = mapped_column()
    avatar_url: Mapped[str | None] = mapped_column()
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    session_tokens = relationship("SessionToken", back_populates="caret_user", cascade="all, delete-orphan")
    billing_subscriptions = relationship("BillingSubscription", back_populates="caret_user", cascade="all, delete-orphan")
    billing_credit_transactions = relationship("BillingCreditTransaction", back_populates="caret_user", cascade="all, delete-orphan")
    billing_invoices = relationship("BillingInvoice", back_populates="caret_user", cascade="all, delete-orphan")
    credit_balances = relationship("CreditBalance", back_populates="caret_user", cascade="all, delete-orphan")
    credit_topups = relationship("CreditTopup", back_populates="caret_user", cascade="all, delete-orphan")
    credit_charges = relationship("CreditCharge", back_populates="caret_user", cascade="all, delete-orphan")
    user = relationship("User", back_populates="caret_user", uselist=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "provider": self.provider,
            "role": self.role,
            "email": self.email,
            "name": self.name,
            "avatar_url": self.avatar_url,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "metadata": self.metadata_,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class BillingPlan(Base):
    """Subscription plans (Lemon variant mapping)."""

    __tablename__ = "billing_plans"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column()
    monthly_credits: Mapped[float] = mapped_column()
    monthly_bonus_credits: Mapped[float] = mapped_column()
    add_amount_usd: Mapped[float] = mapped_column()
    add_bonus_percent: Mapped[float] = mapped_column()
    price_usd: Mapped[float] = mapped_column()
    currency: Mapped[str] = mapped_column(default="USD")
    credits_per_usd: Mapped[float] = mapped_column()
    renew_interval_days: Mapped[int] = mapped_column(default=30)
    lemon_product_id: Mapped[str | None] = mapped_column(nullable=True)
    lemon_variant_id: Mapped[str | None] = mapped_column(nullable=True)
    features: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict)
    active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    subscriptions = relationship("BillingSubscription", back_populates="plan", cascade="all, delete-orphan")
    invoices = relationship("BillingInvoice", back_populates="plan", cascade="all, delete-orphan")
    transactions = relationship("BillingCreditTransaction", back_populates="plan", cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "monthly_credits": self.monthly_credits,
            "price_usd": self.price_usd,
            "currency": self.currency,
            "credits_per_usd": self.credits_per_usd,
            "renew_interval_days": self.renew_interval_days,
            "lemon_product_id": self.lemon_product_id,
            "lemon_variant_id": self.lemon_variant_id,
            "features": self.features or {},
            "active": self.active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class BillingSubscription(Base):
    """User subscription snapshot (Lemon subscription mapping)."""

    __tablename__ = "billing_subscriptions"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("caret_users.user_id", ondelete="CASCADE"), index=True)
    plan_id: Mapped[str] = mapped_column(ForeignKey("billing_plans.id"))
    status: Mapped[str] = mapped_column()
    start_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    renew_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ends_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    transaction_id: Mapped[str | None] = mapped_column(ForeignKey("billing_credit_transactions.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    plan = relationship("BillingPlan", back_populates="subscriptions")
    caret_user = relationship("CaretUser", back_populates="billing_subscriptions")

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "plan_id": self.plan_id,
            "status": self.status,
            "start_at": self.start_at.isoformat() if self.start_at else None,
            "renew_at": self.renew_at.isoformat() if self.renew_at else None,
            "ends_at": self.ends_at.isoformat() if self.ends_at else None,
            "lemon_subscription_id": self.lemon_subscription_id,
            "lemon_customer_id": self.lemon_customer_id,
            "lemon_checkout_id": self.lemon_checkout_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class BillingCreditTransaction(Base):
    """Payment transaction requests/status (checkout/order/subscription)."""

    __tablename__ = "billing_credit_transactions"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("caret_users.user_id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column()
    kind: Mapped[str] = mapped_column()
    amount_usd: Mapped[float | None] = mapped_column(nullable=True)
    credits: Mapped[float | None] = mapped_column(nullable=True)
    credits_per_usd: Mapped[float | None] = mapped_column(nullable=True)
    lemon_checkout_id: Mapped[str | None] = mapped_column(nullable=True, unique=True)
    lemon_order_id: Mapped[str | None] = mapped_column(nullable=True, unique=True)
    lemon_subscription_id: Mapped[str | None] = mapped_column(nullable=True)
    plan_id: Mapped[str | None] = mapped_column(ForeignKey("billing_plans.id"), nullable=True)
    currency: Mapped[str] = mapped_column()
    gift_credits: Mapped[float | None] = mapped_column(nullable=True)
    checkout_url: Mapped[str | None] = mapped_column(nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    request_payload: Mapped[dict[str, Any] | None] = mapped_column("request_payload", JSON, default=dict)
    event_payload: Mapped[dict[str, Any] | None] = mapped_column("event_payload", JSON, default=dict)
    failure_reason: Mapped[str | None] = mapped_column(nullable=True)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    caret_user = relationship("CaretUser", back_populates="billing_credit_transactions")
    plan = relationship("BillingPlan", back_populates="transactions")
    invoices = relationship("BillingInvoice", back_populates="transaction", cascade="all, delete-orphan")
    topups = relationship("CreditTopup", back_populates="transaction", cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "status": self.status,
            "kind": self.kind,
            "amount_usd": self.amount_usd,
            "credits": self.credits,
            "credits_per_usd": self.credits_per_usd,
            "lemon_checkout_id": self.lemon_checkout_id,
            "lemon_order_id": self.lemon_order_id,
            "lemon_subscription_id": self.lemon_subscription_id,
            "plan_id": self.plan_id,
            "metadata": self.metadata_,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class BillingInvoice(Base):
    """Invoice/receipt snapshot (amount, tax, credits)."""

    __tablename__ = "billing_invoices"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("caret_users.user_id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column()
    transaction_id: Mapped[str | None] = mapped_column(ForeignKey("billing_credit_transactions.id"), nullable=True)
    order_id: Mapped[str | None] = mapped_column(unique=True, nullable=True)
    subscription_id: Mapped[str | None] = mapped_column(nullable=True, index=True)
    order_number: Mapped[str | None] = mapped_column(nullable=True)
    plan_id: Mapped[str | None] = mapped_column(ForeignKey("billing_plans.id"), nullable=True)
    status: Mapped[str] = mapped_column()
    total: Mapped[float | None] = mapped_column(nullable=True)
    tax: Mapped[float | None] = mapped_column(nullable=True)
    tax_name: Mapped[str | None] = mapped_column(nullable=True)
    tax_rate: Mapped[float | None] = mapped_column(nullable=True)
    credits: Mapped[float | None] = mapped_column(nullable=True)
    currency: Mapped[str] = mapped_column(default="USD")
    currency_rate: Mapped[float | None] = mapped_column(nullable=True)
    card_brand: Mapped[str | None] = mapped_column(nullable=True)
    card_last_four: Mapped[str | None] = mapped_column(nullable=True)
    user_name: Mapped[str | None] = mapped_column(nullable=True)
    user_email: Mapped[str | None] = mapped_column(nullable=True)
    renews_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    transaction = relationship("BillingCreditTransaction", back_populates="invoices")
    plan = relationship("BillingPlan", back_populates="invoices")
    caret_user = relationship("CaretUser", back_populates="billing_invoices")

    topups = relationship("CreditTopup", back_populates="invoice", cascade="all, delete-orphan")
    charges = relationship("CreditCharge", back_populates="invoice", cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "kind": self.kind,
            "transaction_id": self.transaction_id,
            "order_id": self.order_id,
            "subscription_id": self.subscription_id,
            "order_number": self.order_number,
            "plan_id": self.plan_id,
            "status": self.status,
            "total": self.total,
            "tax": self.tax,
            "credits": self.credits,
        "currency": self.currency,
        "card_brand": self.card_brand,
        "card_last_four": self.card_last_four,
        "user_name": self.user_name,
        "user_email": self.user_email,
        "renews_at": self.renews_at.isoformat() if self.renews_at else None,
            "metadata": self.metadata_,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class BillingWebhookEvent(Base):
    """Raw webhook events storage (e.g., Lemon Squeezy)."""

    __tablename__ = "billing_webhook_events"

    id: Mapped[str] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column()
    event_type: Mapped[str] = mapped_column()
    signature_valid: Mapped[bool] = mapped_column(default=False)
    payload: Mapped[bytes | None] = mapped_column()
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "provider": self.provider,
            "event_type": self.event_type,
            "signature_valid": self.signature_valid,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class CreditBalance(Base):
    """Credit balance pool (subscription/bonus/add_on)."""

    __tablename__ = "credit_balances"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("caret_users.user_id", ondelete="CASCADE"), index=True)
    pool_type: Mapped[str] = mapped_column(index=True)
    source_id: Mapped[str | None] = mapped_column(nullable=True)
    amount: Mapped[float] = mapped_column(default=0.0)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    priority: Mapped[int] = mapped_column(default=1, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)

    charges = relationship("CreditCharge", back_populates="balance", cascade="all, delete-orphan")
    caret_user = relationship("CaretUser", back_populates="credit_balances")

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "pool_type": self.pool_type,
            "source_id": self.source_id,
            "amount": self.amount,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "priority": self.priority,
            "metadata": self.metadata_,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class CreditTopup(Base):
    """Credit top-ups (subscription renewal, bonus, add-on purchase, adjustment)."""

    __tablename__ = "credit_topups"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("caret_users.user_id", ondelete="CASCADE"), index=True)
    pool_type: Mapped[str] = mapped_column(index=True)
    amount: Mapped[float] = mapped_column(default=0.0)
    amount_usd: Mapped[float | None] = mapped_column(nullable=True)
    credits_per_usd: Mapped[float | None] = mapped_column(nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source: Mapped[str] = mapped_column(index=True)
    transaction_id: Mapped[str | None] = mapped_column(ForeignKey("billing_credit_transactions.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)

    transaction = relationship("BillingCreditTransaction", back_populates="topups")
    caret_user = relationship("CaretUser", back_populates="credit_topups")

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "pool_type": self.pool_type,
            "amount": self.amount,
            "amount_usd": self.amount_usd,
            "credits_per_usd": self.credits_per_usd,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "source": self.source,
            "transaction_id": self.transaction_id,
            "metadata": self.metadata_,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class CreditCharge(Base):
    """Credit deductions/refunds. Refunds use negative debited_amount."""

    __tablename__ = "credit_charges"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("caret_users.user_id", ondelete="CASCADE"), index=True)
    usage_id: Mapped[str | None] = mapped_column(nullable=True, index=True)
    pool_id: Mapped[str] = mapped_column(ForeignKey("credit_balances.id", ondelete="CASCADE"), index=True)
    debited_amount: Mapped[float] = mapped_column()  # refunds/adjustments can be negative
    cost_usd: Mapped[float | None] = mapped_column(nullable=True)
    credits_per_usd: Mapped[float | None] = mapped_column(nullable=True)
    model_key: Mapped[str | None] = mapped_column(nullable=True)
    unit_price_usd: Mapped[float | None] = mapped_column(nullable=True)
    invoice_id: Mapped[str | None] = mapped_column(ForeignKey("billing_invoices.id", ondelete="SET NULL"), nullable=True, index=True)
    currency: Mapped[str] = mapped_column(default="USD")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)

    invoice = relationship("BillingInvoice", back_populates="charges")
    balance = relationship("CreditBalance", back_populates="charges")
    caret_user = relationship("CaretUser", back_populates="credit_charges")

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "usage_id": self.usage_id,
            "pool_id": self.pool_id,
            "debited_amount": self.debited_amount,
            "cost_usd": self.cost_usd,
            "credits_per_usd": self.credits_per_usd,
            "model_key": self.model_key,
            "unit_price_usd": self.unit_price_usd,
            "invoice_id": self.invoice_id,
            "currency": self.currency,
            "metadata": self.metadata_,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
