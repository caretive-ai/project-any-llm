"""Add caret_users and session_tokens tables.

Revision ID: c3d7e2f1c6f0
Revises: e7c85cc73bfa
Create Date: 2025-11-23 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d7e2f1c6f0"
down_revision: str | Sequence[str] | None = "e7c85cc73bfa"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "caret_users",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("avatar_url", sa.String(), nullable=True),
        sa.Column("refresh_token", sa.String(), nullable=True),
        sa.Column("access_token_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", name="uq_caret_users_user"),
    )
    op.create_index(op.f("ix_caret_users_provider"), "caret_users", ["provider"], unique=False)
    op.create_index(op.f("ix_caret_users_user_id"), "caret_users", ["user_id"], unique=False)

    op.create_table(
        "session_tokens",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("api_key_id", sa.String(), nullable=False),
        sa.Column("refresh_token_hash", sa.String(), nullable=False),
        sa.Column("refresh_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("refresh_token_hash", name="uq_session_tokens_refresh_hash"),
    )
    op.create_index(op.f("ix_session_tokens_api_key_id"), "session_tokens", ["api_key_id"], unique=False)
    op.create_index(op.f("ix_session_tokens_refresh_token_hash"), "session_tokens", ["refresh_token_hash"], unique=False)
    op.create_index(op.f("ix_session_tokens_user_id"), "session_tokens", ["user_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_session_tokens_user_id"), table_name="session_tokens")
    op.drop_index(op.f("ix_session_tokens_refresh_token_hash"), table_name="session_tokens")
    op.drop_index(op.f("ix_session_tokens_api_key_id"), table_name="session_tokens")
    op.drop_table("session_tokens")

    op.drop_index(op.f("ix_caret_users_user_id"), table_name="caret_users")
    op.drop_index(op.f("ix_caret_users_provider"), table_name="caret_users")
    op.drop_table("caret_users")
