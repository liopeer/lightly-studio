"""Resolve an embedding-plot region (a polygon of 2D vertices) to the sample ids it encloses.

Given a polygon and a collection, load the cached, deterministic 2D projection of that
collection's embeddings and return the ids of the samples whose coordinates fall inside the
polygon. Taking the geometry (a handful of vertices) rather than an explicit id list keeps the
input constant-size regardless of how many samples the region covers.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
from numpy.typing import NDArray
from sqlmodel import Session

from lightly_studio.models.embedding_region import EmbeddingRegion
from lightly_studio.resolvers import embedding_model_resolver, twodim_embedding_resolver


def get_sample_ids_in_region(
    session: Session,
    collection_id: UUID,
    region: EmbeddingRegion,
) -> list[UUID]:
    """Return the sample ids whose cached 2D coordinates fall inside ``region``."""
    embedding_model = (
        embedding_model_resolver.get_by_id(
            session=session, embedding_model_id=region.embedding_model_id
        )
        if region.embedding_model_id is not None
        else embedding_model_resolver.get_latest_complete_by_collection_id(
            session=session,
            collection_id=collection_id,
        )
    )
    if embedding_model is not None and embedding_model.collection_id != collection_id:
        raise ValueError("Embedding model does not belong to this collection.")
    if embedding_model is None:
        return []
    if not embedding_model_resolver.is_complete_for_collection(
        session=session,
        collection_id=collection_id,
        embedding_model_id=embedding_model.embedding_model_id,
    ):
        raise ValueError("Selected embedding model does not cover the complete collection.")

    x_array, y_array, sample_ids = twodim_embedding_resolver.get_twodim_embeddings(
        session=session,
        collection_id=collection_id,
        embedding_model_id=embedding_model.embedding_model_id,
    )
    if len(sample_ids) == 0:
        return []

    inside_mask = _points_in_polygon(x=x_array, y=y_array, region=region)
    return [sample_id for sample_id, inside in zip(sample_ids, inside_mask) if inside]


def _points_in_polygon(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    region: EmbeddingRegion,
) -> NDArray[np.bool_]:
    """Vectorized even-odd ray-casting point-in-polygon test.

    A ray is cast to the right (+x); an edge is counted when exactly one endpoint is strictly
    above the point's y. This intentionally reproduces the frontend's ``isPointInPolygon`` so
    that a region selected in the plot maps to the same set of samples on the server:
    https://github.com/lightly-ai/lightly-studio/blob/7c44af936d1193a0fcedf91644bf91f5c9e9ef55/lightly_studio_view/src/lib/components/PlotPanel/isPointInPolygon/isPointInPolygon.ts#L18-L49
    """
    vertices_x = np.asarray([vertex.x for vertex in region.polygon], dtype=np.float32)
    vertices_y = np.asarray([vertex.y for vertex in region.polygon], dtype=np.float32)

    px = np.asarray(x, dtype=np.float32)
    py = np.asarray(y, dtype=np.float32)

    inside = np.zeros(px.shape, dtype=np.bool_)

    # Previous vertex of each edge: (vertices[j], vertices[i]) with j = i - 1 (wrapping).
    prev_x = np.roll(vertices_x, 1)
    prev_y = np.roll(vertices_y, 1)

    for xi, yi, xj, yj in zip(vertices_x, vertices_y, prev_x, prev_y):
        # Edge straddles the horizontal ray at py (one endpoint strictly above, one not).
        straddles = (yi > py) != (yj > py)
        denominator = yj - yi
        # Guard against a horizontal edge (denominator == 0); those never straddle, so the
        # value used here is irrelevant, but avoid dividing by zero.
        safe_denominator = np.where(denominator == 0, 1.0, denominator)
        intersection_x = (xj - xi) * (py - yi) / safe_denominator + xi
        crosses = straddles & (px < intersection_x)
        inside ^= crosses

    return inside
