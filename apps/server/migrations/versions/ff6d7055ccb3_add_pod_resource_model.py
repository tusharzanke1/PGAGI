"""add pod & resource model

Revision ID: ff6d7055ccb3
Revises: f1d5bc37bceb
Create Date: 2024-06-06 13:16:13.521912

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'ff6d7055ccb3'
down_revision: Union[str, None] = 'f1d5bc37bceb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('resource',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.Column('display_name', sa.String(length=255), nullable=False),
    sa.Column('type', sa.Enum('cpu', 'gpu', name='resource_type'), nullable=False),
    sa.Column('category', sa.String(), nullable=True),
    sa.Column('ram', sa.Numeric(precision=5, scale=2), nullable=True),
    sa.Column('secure_price', sa.Numeric(precision=5, scale=2), nullable=True),
    sa.Column('one_month_price', sa.Numeric(precision=5, scale=2), nullable=True),
    sa.Column('three_month_price', sa.Numeric(precision=5, scale=2), nullable=True),
    sa.Column('six_month_price', sa.Numeric(precision=5, scale=2), nullable=True),
    sa.Column('max_gpu', sa.Numeric(precision=5, scale=2), nullable=True),
    sa.Column('lowest_price', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('status', sa.Enum('low', 'high', 'unavailable', name='status_enum'), nullable=True),
    sa.Column('disc_type', sa.Enum('ssd', 'nvme', 'unavailable', name='disc_type_enum'), nullable=True),
    sa.Column('cloud_type', sa.Enum('secure cloud', 'community cloud', 'unavailable', name='cloud_type_enum'), nullable=True),
    sa.Column('region', sa.String(length=255), nullable=False),
    sa.Column('cuda_version', sa.Numeric(precision=5, scale=2), nullable=True),
    sa.Column('is_deleted', sa.Boolean(), nullable=True),
    sa.Column('created_on', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_on', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_resource_is_deleted'), 'resource', ['is_deleted'], unique=False)
    op.create_table('pod',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('pod_name', sa.String(), nullable=True),
    sa.Column('price', sa.Numeric(precision=5, scale=2), nullable=True),
    sa.Column('status', sa.Enum('running', 'stopped', name='status_enum'), nullable=True),
    sa.Column('provider', sa.String(), nullable=True),
    sa.Column('category', sa.String(), nullable=True),
    sa.Column('type', sa.Enum('cpu', 'gpu', name='category_enum'), nullable=True),
    sa.Column('gpu_count', sa.Numeric(precision=5, scale=2), nullable=True),
    sa.Column('isinstance_pricing', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('is_deleted', sa.Boolean(), nullable=True),
    sa.Column('account_id', sa.UUID(), nullable=True),
    sa.Column('created_by', sa.UUID(), nullable=True),
    sa.Column('modified_by', sa.UUID(), nullable=True),
    sa.Column('created_on', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_on', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['account_id'], ['account.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['created_by'], ['user.id'], name='fk_created_by', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['modified_by'], ['user.id'], name='fk_modified_by', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_pod_created_by'), 'pod', ['created_by'], unique=False)
    op.create_index(op.f('ix_pod_is_deleted'), 'pod', ['is_deleted'], unique=False)
    op.create_index(op.f('ix_pod_modified_by'), 'pod', ['modified_by'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_pod_modified_by'), table_name='pod')
    op.drop_index(op.f('ix_pod_is_deleted'), table_name='pod')
    op.drop_index(op.f('ix_pod_created_by'), table_name='pod')
    op.drop_table('pod')
    op.drop_index(op.f('ix_resource_is_deleted'), table_name='resource')
    op.drop_table('resource')
    # ### end Alembic commands ###
