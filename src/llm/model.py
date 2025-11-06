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
    tag: str
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
            HYBRID_DEEPSEEK_QWEN_7B = ModelSpec(
                name="Deepseek-R1-distill-qwen-7B-Q8_0",
                tag="deepseek_qwen_7b",
                location=Location.LOCAL,
                backend=Backend.LLAMACPP,
                agent_backend=AgentBackend.LLAMACPPAGENT,
                kind=ModelKind.HYBRID,
                source=SourceHuggingface(
                    repo_id="bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
                    filename="DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf",
                    local_dir="data/model/"
                    #https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/blob/main/DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf
                ),
            )

            THINK_PHI4_MINI_3_8B = ModelSpec(
                name="Phi-4-mini-reasoning-3.8B",
                tag="phi4_mini_reasoning_3.8b",
                location=Location.LOCAL,
                backend=Backend.LLAMACPP,
                agent_backend=AgentBackend.LLAMACPPAGENT,
                kind=ModelKind.HYBRID,
                source=SourceHuggingface(
                    repo_id="unsloth/Phi-4-mini-reasoning-GGUF",
                    filename="Phi-4-mini-reasoning-BF16.gguf",
                    local_dir="data/model/"
                    #https://huggingface.co/unsloth/Phi-4-mini-reasoning-GGUF
                ),
            )

        class Ollama(Enum):
            class Qwen3(Enum):
                CODER_30B = ModelSpec(
                    name="Qwen3-Coder-30B",
                    tag="qwen3_coder_30b",
                    location=Location.LOCAL,
                    backend=Backend.OLLAMA,
                    agent_backend=AgentBackend.LANGCHAIN,
                    kind=ModelKind.HYBRID,
                    source=SourceOllama(
                        model_id="qwen3-coder:30b",
                        #https://ollama.com/library/qwen3
                    ),
                )

                VANILLA_30B = ModelSpec(
                    name="Qwen3-30B",
                    tag="qwen3_30b",
                    location=Location.LOCAL,
                    backend=Backend.OLLAMA,
                    agent_backend=AgentBackend.LANGCHAIN,
                    kind=ModelKind.HYBRID,
                    source=SourceOllama(
                        model_id="qwen3:30b",
                        #https://ollama.com/library/qwen3
                    ),
                )

                VANILLA_14B = ModelSpec(
                    name="Qwen3-14B",
                    tag="qwen3_14b",
                    location=Location.LOCAL,
                    backend=Backend.OLLAMA,
                    agent_backend=AgentBackend.LANGCHAIN,
                    kind=ModelKind.HYBRID,
                    source=SourceOllama(
                        model_id="qwen3:14b",
                        #https://ollama.com/library/qwen3
                    ),
                )

                VANILLA_8B = ModelSpec(
                    name="Qwen3-8B",
                    tag="qwen3_8b",
                    location=Location.LOCAL,
                    backend=Backend.OLLAMA,
                    agent_backend=AgentBackend.LANGCHAIN,
                    kind=ModelKind.HYBRID,
                    source=SourceOllama(
                        model_id="qwen3:8b",
                        #https://ollama.com/library/qwen3
                    ),
                )

                VANILLA_4B = ModelSpec(
                    name="Qwen3-4B",
                    tag="qwen3_4b",
                    location=Location.LOCAL,
                    backend=Backend.OLLAMA,
                    agent_backend=AgentBackend.LANGCHAIN,
                    kind=ModelKind.HYBRID,
                    source=SourceOllama(
                        model_id="qwen3:4b",
                        #https://ollama.com/library/qwen3
                    ),
                )

                INSTRUCT_30B = ModelSpec(
                    name="Qwen3-Instruct-30B",
                    tag="qwen3_instruct_30b",
                    location=Location.LOCAL,
                    backend=Backend.OLLAMA,
                    agent_backend=AgentBackend.LANGCHAIN,
                    kind=ModelKind.HYBRID,
                    source=SourceOllama(
                        model_id="qwen3:30b-instruct",
                        #https://ollama.com/library/qwen3/tags
                    ),
                )

            class Llama(Enum):
                NEMOTRON_8B = ModelSpec(
                    name="Llama3.1-nemotron-8B",
                    tag="llama_nemotron_8b",
                    location=Location.LOCAL,
                    backend=Backend.OLLAMA,
                    agent_backend=AgentBackend.LANGCHAIN,
                    kind=ModelKind.HYBRID,
                    source=SourceOllama(
                        model_id="Randomblock1/nemotron-nano:8b",
                        #https://https://ollama.com/Randomblock1/nemotron-nano
                    ),
                )

            HYBRID_PHI4_MINI_3_8B = ModelSpec(
                name="Phi-4-mini-3.8B",
                tag="phi4_mini_3.8b",
                location=Location.LOCAL,
                backend=Backend.OLLAMA,
                agent_backend=AgentBackend.LANGCHAIN,
                kind=ModelKind.HYBRID,
                source=SourceOllama(
                    model_id="phi4-mini",
                    #https://ollama.com/library/phi4-mini
                ),
            )

            HYBRID_GPT_OSS_20B = ModelSpec(
                name="GPT-OSS-20B",
                tag="gpt_oss_20b",
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
            # institut
            GPT_OSS_20B = ModelSpec(
                name="GPT-OSS-20B",
                tag="gpt_oss_20b",
                location=Location.REMOTE,
                backend=Backend.OTHER,
                agent_backend=AgentBackend.LANGCHAIN,
                kind=ModelKind.HYBRID,
                source=SourceRemote(
                    endpoint_url="",
                    model_id="",
                ),
            )

            # institut
            PHI4_PLUS = ModelSpec(
                name="Phi-4-reasoning-plus-14B" ,
                tag="phi4_plus_14b",
                location=Location.REMOTE,
                backend=Backend.OTHER,
                agent_backend=AgentBackend.LANGCHAIN,
                kind=ModelKind.HYBRID,
                source=SourceRemote(
                    endpoint_url="",
                    model_id="",
                ),
            )

            GPT4 = ModelSpec(
                name="OpenAI-GPT-4",
                tag="gpt4",
                location=Location.REMOTE,
                backend=Backend.OTHER,
                agent_backend=AgentBackend.LANGCHAIN,
                kind=ModelKind.HYBRID,
                source=SourceRemote(
                    endpoint_url="https://api.openai.com/v1",
                    model_id="gpt-4",
                ),
            )

            GPT5 = ModelSpec(
                name="OpenAI-GPT-5",
                tag="gpt5",
                location=Location.REMOTE,
                backend=Backend.OTHER,
                agent_backend=AgentBackend.LANGCHAIN,
                kind=ModelKind.HYBRID,
                source=SourceRemote(
                    endpoint_url="https://api.openai.com/v1",
                    model_id="gpt-5",
                ),
            )

    class Embedding(Enum):
        QWEN3_8B = ModelSpec(
                name="Qwen3-Embedding-8B",
                tag="",
                location=Location.LOCAL,
                backend=Backend.OTHER,
                agent_backend=AgentBackend.OTHER,
                kind=ModelKind.EMBEDDING,
                source=SourceHuggingface(
                    repo_id="onathanMiddleton/Qwen3-Embedding-8B-GGUF",
                    filename="Qwen3-Embedding-8B-BF16.gguf",
                    local_dir="data/embeddings/"
                    #https://huggingface.co/JonathanMiddleton/Qwen3-Embedding-8B-GGUF
                ),
            )
