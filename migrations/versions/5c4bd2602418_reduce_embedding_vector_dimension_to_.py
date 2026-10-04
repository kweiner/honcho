"""reduce_embedding_vector_dimension_to_1536

The upstream embedding provider (qwen/qwen3-embedding-8b via OpenRouter)
stopped accepting arbitrary "dimensions" values; only fixed Matryoshka
checkpoints (1024/1536/2048/4096) are valid, and pgvector's HNSW index has a
hard 2000-dimension ceiling on the plain vector type, so 2048+ is not an
option. This migration moves message_embeddings.embedding and
documents.embedding from vector(2000) to vector(1536).

Existing embedding values cannot be resized in place, so they are nulled out
and sync_state is reset to 'pending' so the reconciler (or a one-off
backfill) re-embeds everything at the new dimension. Postgres/pgvector
rebuilds the HNSW index automatically as part of ALTER COLUMN TYPE (verified
directly against the target Postgres/pgvector version before writing this),
so no explicit drop/recreate of the index is needed.

Revision ID: 5c4bd2602418
Revises: e4eba9cfaa6f
Create Date: 2026-07-29

"""

from collections.abc import Sequence

from alembic import op
from pgvector.sqlalchemy import Vector

from migrations.utils import get_schema

# revision identifiers, used by Alembic.
revision: str = "5c4bd2602418"
down_revision: str | None = "a7c3e9f1b2d4"  # rebased onto v3.2.0: upstream's
# add_document_sources_table also descends from e4eba9cfaa6f, which would leave
# Alembic with two heads. Chain after it instead.
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()

OLD_DIM = 2000
NEW_DIM = 1536


def _resize(old_dim: int, new_dim: int) -> None:
    # Existing vectors are the wrong dimensionality for the new column type
    # and cannot be cast; clear them and mark for re-embedding.
    op.execute(
        f"UPDATE {schema}.message_embeddings "
        "SET embedding = NULL, sync_state = 'pending', sync_attempts = 0, last_sync_at = NULL"
    )
    op.execute(
        f"UPDATE {schema}.documents "
        "SET embedding = NULL, sync_state = 'pending', sync_attempts = 0, last_sync_at = NULL"
    )

    op.alter_column(
        "message_embeddings",
        "embedding",
        existing_type=Vector(old_dim),
        type_=Vector(new_dim),
        schema=schema,
    )
    op.alter_column(
        "documents",
        "embedding",
        existing_type=Vector(old_dim),
        type_=Vector(new_dim),
        schema=schema,
    )


def upgrade() -> None:
    """Null out + resize embedding columns from 2000 to 1536 dimensions."""
    _resize(OLD_DIM, NEW_DIM)


def downgrade() -> None:
    """Revert embedding columns to vector(2000); existing 1536-dim data is nulled."""
    _resize(NEW_DIM, OLD_DIM)
