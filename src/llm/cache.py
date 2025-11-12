from pathlib import Path
from typing import Dict, Union

from langchain_ollama import ChatOllama
from llama_cpp import Llama

import config
import debug
from llm.model import Backend, Model, ModelSpec, SourceFile, SourceHuggingface, SourceLink
from llm.prepare import prepare_model_source
from util import console

class Cache():
    _instances: Dict[ModelSpec, Union[Llama, ChatOllama]] = {}

    def get(model: Model):
        spec = model.value

        if spec not in Cache._instances:
            prepare_model_source(model)

            if spec.backend == Backend.LLAMACPP:
                src = model.value.source

                if isinstance(src, SourceFile):
                    path = src.path
                elif isinstance(src, SourceLink):
                    path = src.path
                elif isinstance(src, SourceHuggingface):
                    path = src.local_dir + src.filename
                else:
                    raise ValueError("Unsupported LlamaCppProvider Source")

                if not Path(path).exists():
                    raise FileNotFoundError(f"Cant load model: file not found: {path}")

                try:
                    llm = Cache._create_llama(path)
                except:
                    Cache._instances.clear()
                    llm = Cache._create_llama(path)

            elif spec.backend == Backend.OLLAMA:
                llm = ChatOllama(
                    model=spec.source.model_id,
                    verbose=debug.VERBOSE_OLLAMA,
                    seed=config.ACTIVE_CONFIG.seed,
                    temperature=config.ACTIVE_CONFIG.temperature,
                )
            else:
                raise ValueError(f"Unknown backend: {spec.backend}")
            
            console.pretty(
                console.banner(f"[LLM STARTED] {model!s}", color=console.Color.GREEN),
            )

            Cache._instances[spec] = llm

        return Cache._instances[spec]
    
    def _create_llama(path: str) -> Llama:
        llm = Llama(
            model_path=path,
            n_gpu_layers=config.Backend._n_gpu_layers,
            n_threads=config.Backend._n_threads,
            n_ctx=config.Backend._n_context,
            verbose=debug.VERBOSE_LLAMACPP,
            seed=config.ACTIVE_CONFIG.seed,
            temperature=config.ACTIVE_CONFIG.temperature,
        )
        
        return llm