from pathlib import Path
import requests
import ollama
from huggingface_hub import hf_hub_download

from llm.model import (
    Model,
    SourceFile,
    SourceLink,
    SourceHuggingface,
    SourceOllama,
    SourceRemote
)

def prepare_model_source(model: Model) -> None:
    """
    Ensure the model file is locally available if needed.
    """
    src = model.value.source

    # nothing to download
    if isinstance(src, SourceRemote):
        return
    
    # download from ollama to ollama store
    if isinstance(src, SourceOllama):
        ollama.pull(model=src.model_id)
        return
    
    # download from huggingface to local storage
    if isinstance(src, SourceHuggingface):
        hf_hub_download(
            repo_id=src.repo_id,
            filename=src.filename,
            local_dir=src.local_dir,
        )
        return
    
    ### FILE ###

    # path already present
    if isinstance(src, SourceFile):
        local_file = Path(src.path)
        if not local_file.exists():
            raise FileNotFoundError(f"Model file not found: {local_file}")
        return

    # download once to temp dir
    if isinstance(src, SourceLink):
        local_file = Path(src.path)

        if not local_file.exists():
            try:
                # connection test
                resp = requests.get(src.url, stream=True, timeout=(10, 60))
                resp.raise_for_status()
            except requests.exceptions.RequestException as e:
                # z.B. DNS-Fehler, 404, Timeout …
                raise RuntimeError(f"Failed to download model from {src.url}: {e}") from e

            try:
                local_file.parent.mkdir(parents=True, exist_ok=True)
                with open(local_file, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:  # leere Chunks ignorieren
                            f.write(chunk)
            except OSError as e:
                raise RuntimeError(f"Could not write downloaded model to {local_file}: {e}") from e

            if not local_file.exists() or local_file.stat().st_size == 0:
                raise RuntimeError(f"Downloaded file is missing or empty: {local_file}")
        return

    raise ValueError(f"Unknown source type: {type(src)}")