"""Utility functions for embedding-related operations in Lightly Studio datasets."""

from uuid import UUID

from sqlmodel import Session

from lightly_studio.resolvers import embedding_model_resolver


def collection_has_embeddings(session: Session, collection_id: UUID) -> bool:
    """Check if there are any embeddings available for the given collection.

    Args:
        session: Database session for resolver operations.
        collection_id: The ID of the collection to check for embeddings.

    Returns:
        True if embeddings exist for the collection, False otherwise.
    """
    embedding_model = embedding_model_resolver.get_latest_complete_by_collection_id(
        session=session, collection_id=collection_id
    )
    return embedding_model is not None
