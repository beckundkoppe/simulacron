from dataclasses import dataclass
from enum import Enum
from typing import Union

class Backend(Enum):
    LLAMACPP    = "llamacpp",
    OLLAMA      = "ollama",
    OTHER       = "other",

class AgentBackend(Enum):
    LLAMACPPAGENT   = "llamacppagent"   # llama.cpp & llama_cpp_agent
    LANGCHAIN       = "langchain"       # llama.cpp & langchain
    OTHER           = "other"

class Location(Enum):
    LOCAL   = "local"   # runs on local maschine
    REMOTE  = "remote"  # runs on remote server

@dataclass
class SourceFile:
    path: str

@dataclass
class SourceLink:
    url: str
    path: str

@dataclass
class SourceHuggingface:
    repo_id: str
    filename: str
    local_dir: str

@dataclass
class SourceOllama:
    model_id: str

@dataclass
class SourceRemote:
    endpoint_url: str
    model_id: str

class ModelKind(Enum):
    INSTRUCT        = "instruct"
    TOOL            = "tool"
    HYBRID          = "hybrid"
    EMBEDDING       = "embedding"

@dataclass
class ModelSpec:
    name: str
    location: Location
    backend: Backend
    agent_backend: AgentBackend
    kind: ModelKind
    source: Union[SourceFile, SourceLink, SourceHuggingface, SourceRemote]

class Model(Enum):
    # ----------------------------------------------------------
    # Local models
    # ----------------------------------------------------------
    class Local(Enum):
        class LlamaCpp(Enum):
            INSTRUCT_MISTRAL_7B = ModelSpec(
                name="Mistral-7B-Q4-Instruct",
                location=Location.LOCAL,
                backend=Backend.LLAMACPP,
                agent_backend=AgentBackend.LLAMACPPAGENT,
                kind=ModelKind.INSTRUCT,
                source=SourceLink(
                    url="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                    path="data/model/mistral_7b_instruct.gguf"
                ),
            )

            TOOL_QWEN3_4B = ModelSpec(
                name="Qwen3-4B-Toolcalling",
                location=Location.LOCAL,
                backend=Backend.LLAMACPP,
                agent_backend=AgentBackend.LLAMACPPAGENT,
                kind=ModelKind.TOOL,
                source=SourceHuggingface(
                    repo_id="Manojb/Qwen3-4B-toolcalling-gguf-codex",
                    filename="Qwen3-4B-Function-Calling-Pro.gguf",
                    local_dir="data/model/"
                    #https://huggingface.co/Manojb/Qwen3-4B-toolcalling-gguf-codex
                ),
            )

            HYBRID_LLAMA3_GROQ_8B_Q8 = ModelSpec(
                name="Llama3-Groq-8B-Q8-Toolcalling",
                location=Location.LOCAL,
                backend=Backend.LLAMACPP,
                agent_backend=AgentBackend.LLAMACPPAGENT,
                kind=ModelKind.HYBRID,
                source=SourceHuggingface(
                    repo_id="rumbleFTW/Llama-3-Groq-8B-Tool-Use-Q8_0-GGUF",
                    filename="llama-3-groq-8b-tool-use-q8_0.gguf",
                    local_dir="data/model/"
                    #https://huggingface.co/rumbleFTW/Llama-3-Groq-8B-Tool-Use-Q8_0-GGUF
                ),
            )

            HYBRID_QWEN3_14B_BF16 = ModelSpec(
                name="Qwen3-14B-Thinking",
                location=Location.LOCAL,
                backend=Backend.LLAMACPP,
                agent_backend=AgentBackend.LLAMACPPAGENT,
                source=SourceHuggingface(
                    repo_id="unsloth/Qwen3-14B-GGUF",
                    filename="Qwen3-14B-BF16.gguf",
                    local_dir="data/model/"
                    #https://huggingface.co/unsloth/Qwen3-14B-GGUF
                ),
                kind=ModelKind.HYBRID,
            )

            HYBRID_QWEN3_14B_Q6 = ModelSpec(
                name="Qwen3-14B-Q6-Thinking",
                location=Location.LOCAL,
                backend=Backend.LLAMACPP,
                agent_backend=AgentBackend.LLAMACPPAGENT,
                source=SourceHuggingface(
                    repo_id="unsloth/Qwen3-14B-GGUF",
                    filename="Qwen3-14B-Q6_K.gguf",
                    local_dir="data/model/"
                    #https://huggingface.co/unsloth/Qwen3-14B-GGUF
                ),
                kind=ModelKind.HYBRID,
            )

        class Ollama(Enum):
            HYBRID_LLAMA3_2_3B = ModelSpec(
                name="Llama3.2-3B",
                location=Location.LOCAL,
                backend=Backend.OLLAMA,
                agent_backend=AgentBackend.LANGCHAIN,
                kind=ModelKind.HYBRID,
                source=SourceOllama(
                    model_id="llama3.2:3b",
                    #https://ollama.com/library/llama3.2
                ),
            )

            INSTRUCT_DEEPSEEK_R1_8B = ModelSpec(
                name="Deepseek-r1-8B",
                location=Location.LOCAL,
                backend=Backend.OLLAMA,
                agent_backend=AgentBackend.LANGCHAIN,
                kind=ModelKind.INSTRUCT,
                source=SourceOllama(
                    model_id="deepseek-r1:8b",
                    #https://ollama.com/library/llama3.2
                ),
            )

            INSTRUCT_GEMMA3_4B = ModelSpec(
                name="Gemma3-4B",
                location=Location.LOCAL,
                backend=Backend.OLLAMA,
                agent_backend=AgentBackend.LANGCHAIN,
                kind=ModelKind.INSTRUCT,
                source=SourceOllama(
                    model_id="gemma3:4b",
                    #https://ollama.com/library/llama3.2
                ),
            )

            HYBRID_GPT_OSS_20B = ModelSpec(
                name="Llama3.2-3B",
                location=Location.LOCAL,
                backend=Backend.OLLAMA,
                agent_backend=AgentBackend.LANGCHAIN,
                kind=ModelKind.HYBRID,
                source=SourceOllama(
                    model_id="gpt-oss:20b",
                    #https://ollama.com/library/gpt-oss
                ),
            )

    # ----------------------------------------------------------
    # Remote models
    # ----------------------------------------------------------
    class Remote(Enum):
            INSTRUCT_MISTRAL_7B = ModelSpec(
                name="Mistral-7B-Instruct",
                location=Location.REMOTE,
                backend=Backend.OTHER,
                agent_backend=AgentBackend.LLAMACPPAGENT,
                kind=ModelKind.INSTRUCT,
                source=SourceRemote(
                    endpoint_url="http://my-llamacpp-server:8080/v1",
                    model_id="mistral-7b-instruct",
                ),
            )
    
            GPT4 = ModelSpec(
                name="OpenAI GPT-4",
                location=Location.REMOTE,
                backend=Backend.OTHER,
                agent_backend=AgentBackend.LANGCHAIN,
                kind=ModelKind.HYBRID,
                source=SourceRemote(
                    endpoint_url="https://api.openai.com/v1",
                    model_id="gpt-4",
                ),
            )

    class Embedding(Enum):
        NOMIC_V1_5_Q8 = ModelSpec(
                name="nomic-embed-text-v1.5",
                location=Location.LOCAL,
                backend=Backend.OTHER,
                agent_backend=AgentBackend.OTHER,
                kind=ModelKind.EMBEDDING,
                source=SourceHuggingface(
                    repo_id="nomic-ai/nomic-embed-text-v1.5-GGUF",
                    filename="nomic-embed-text-v1.5.Q8_0.gguf",
                    local_dir="data/embeddings/"
                    #https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF
                ),
            )
