"""Module to launch the GUI."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass

import uvicorn

from lightly_studio.api.server import Server
from lightly_studio.database import db_manager
from lightly_studio.resolvers import collection_resolver, sample_resolver

logger = logging.getLogger(__name__)


def start_gui(
    name: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> Server:
    """Launch the web interface for the loaded dataset.

    This call blocks until the server stops.

    Args:
        name: Name of the dataset to load. If None, the most recently created dataset
            in the database will be loaded.
        host: Host to bind the server to. Falls back to LIGHTLY_STUDIO_HOST env var.
        port: Port to bind the server to. Falls back to LIGHTLY_STUDIO_PORT env var.

    Returns:
        The Server instance.
    """
    selected_dataset = _validate_has_samples(name=name)

    server = Server(host=host, port=port)
    uvicorn_server = server.create_uvicorn_server()

    dataset_url = (
        f"{server.url}/datasets/{selected_dataset.collection_id}"
        f"/{selected_dataset.sample_type}"
        f"/{selected_dataset.collection_id}"
    )
    logger.info(f"Open the LightlyStudio GUI under: {dataset_url}")

    _run_uvicorn_server(uvicorn_server)
    return server


@dataclass
class _GuiBackgroundState:
    # Store background execution details so stop can target the right server.
    uvicorn_server: uvicorn.Server
    thread: threading.Thread


_GUI_BACKGROUND_STATE: _GuiBackgroundState | None = None


def start_gui_background(
    name: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> Server:
    """Launch the web interface in a background thread.

    Args:
        name: Name of the dataset to load. If None, the most recently created dataset
            in the database will be loaded.
        host: Host to bind the server to. Falls back to LIGHTLY_STUDIO_HOST env var.
        port: Port to bind the server to. Falls back to LIGHTLY_STUDIO_PORT env var.

    Returns:
        The Server instance.
    """
    global _GUI_BACKGROUND_STATE  # noqa: PLW0603
    # TODO(Malte, 01/26): Handle start when a background server is already running.

    selected_dataset = _validate_has_samples(name=name)

    server = Server(host=host, port=port)
    uvicorn_server = server.create_uvicorn_server()

    thread = threading.Thread(
        target=_run_uvicorn_server,
        args=(uvicorn_server,),
        daemon=True,
        name="lightly-studio-gui",
    )
    state = _GuiBackgroundState(uvicorn_server=uvicorn_server, thread=thread)
    _GUI_BACKGROUND_STATE = state

    dataset_url = (
        f"{server.url}/datasets/{selected_dataset.collection_id}"
        f"/{selected_dataset.sample_type}"
        f"/{selected_dataset.collection_id}"
    )
    logger.info(f"Open the LightlyStudio GUI under: {dataset_url}")

    thread.start()
    # TODO(Malte, 01/26): Wait for server startup and surface background errors.
    return server


def stop_gui_background() -> None:
    """Stop the background GUI server."""
    global _GUI_BACKGROUND_STATE  # noqa: PLW0603
    state = _GUI_BACKGROUND_STATE
    if state is None:
        # TODO(Malte, 01/26): Handle stop when no background server is running.
        return

    state.uvicorn_server.should_exit = True
    state.thread.join()
    _GUI_BACKGROUND_STATE = None
    # TODO(Malte, 01/26): Handle background server shutdown failures.


def _run_uvicorn_server(uvicorn_server: uvicorn.Server) -> None:
    """Start a Uvicorn server, handling notebook event loops."""
    # Notebook environments (Colab/Jupyter) already run an event loop.
    # We do this to support running the app in a notebook environment.
    # Reuse the same server instance so serve/run share the same lifecycle state.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        loop.create_task(uvicorn_server.serve())
        return

    # start the app with connection limits and timeouts
    uvicorn_server.run()


def _validate_has_samples(name: str | None = None):
    """Validate that there are samples in the database before starting GUI.

    Args:
        name: Name of the dataset to validate. If None, the first dataset is used.

    Returns:
        The selected CollectionTable.

    Raises:
        ValueError: If the dataset is not found or has no samples.
    """
    session = db_manager.persistent_session()

    if name is not None:
        collection_id = collection_resolver.get_by_name(
            session=session, name=name, parent_collection_id=None
        )
        if collection_id is None:
            raise ValueError(
                f"Dataset '{name}' not found. Please ensure the dataset exists before "
                "starting the GUI."
            )
        selected_dataset = collection_resolver.get_by_id(
            session=session, collection_id=collection_id
        )
    else:
        datasets = collection_resolver.get_all(session=session, offset=0, limit=1)
        if not datasets:
            raise ValueError(
                "No datasets found. Please load a dataset using Dataset class methods "
                "(e.g., add_images_from_path(), add_samples_from_yolo(), etc.) "
                "before starting the GUI."
            )
        selected_dataset = datasets[0]

    sample_count = sample_resolver.count_by_collection_id(
        session=session, collection_id=selected_dataset.collection_id
    )

    if sample_count == 0:
        raise ValueError(
            f"No images have been indexed for dataset '{selected_dataset.name}'. "
            "Please ensure your dataset contains valid images and try loading again."
        )

    return selected_dataset
