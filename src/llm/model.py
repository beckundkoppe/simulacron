from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

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
    api_key: Optional[str] = None

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
    source: Union[SourceFile, SourceLink, SourceHuggingface, SourceOllama, SourceRemote]

    def __hash__(self):
        return hash((self.name, self.tag, self.backend, self.kind))
    
    def __eq__(self, other):
        if not isinstance(other, ModelSpec):
            return False
        return (
            self.name == other.name
            and self.tag == other.tag
            and self.backend == other.backend
            and self.kind == other.kind
        )

class Model(Enum):
    # ----------------------------------------------------------
    # Local models
    # ----------------------------------------------------------
    class Local(Enum):

        class LlamaCpp(Enum):
            class Deepseek(Enum):
                CODER_V2_16B_Q8 = ModelSpec(
                    name="Deepseek-Coder-V2-16B-Q8",
                    tag="deepseek_coder_16b_q8",
                    location=Location.LOCAL,
                    backend=Backend.LLAMACPP,
                    agent_backend=AgentBackend.LLAMACPPAGENT,
                    kind=ModelKind.HYBRID,
                    source=SourceHuggingface(
                        repo_id="lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
                        filename="DeepSeek-Coder-V2-Lite-Instruct-Q8_0.gguf",
                        local_dir="data/model/"
                        #https://huggingface.co/lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF
                    ),
                )

                CODER_V2_16B_Q4 = ModelSpec(
                    name="Deepseek-Coder-V2-16B-Q4",
                    tag="deepseek_coder_16b_q4",
                    location=Location.LOCAL,
                    backend=Backend.LLAMACPP,
                    agent_backend=AgentBackend.LLAMACPPAGENT,
                    kind=ModelKind.HYBRID,
                    source=SourceHuggingface(
                        repo_id="lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
                        filename="DeepSeek-Coder-V2-Lite-Instruct-IQ4_XS.gguf",
                        local_dir="data/model/"
                        #https://huggingface.co/lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF
                    ),
                )

                R1_QWEN_7B = ModelSpec(
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

            class Qwen3(Enum):
                NEMOTRON_14B_Q8 = ModelSpec(
                    name="Nemotron-Qwen2.4-14B-Q8",
                    tag="nemotron_qwen_14B",
                    location=Location.LOCAL,
                    backend=Backend.LLAMACPP,
                    agent_backend=AgentBackend.LLAMACPPAGENT,
                    kind=ModelKind.HYBRID,
                    source=SourceHuggingface(
                        repo_id="bartowski/nvidia_OpenReasoning-Nemotron-14B-GGUF",
                        filename="nvidia_OpenReasoning-Nemotron-14B-Q8_0.gguf",
                        local_dir="data/model/"
                        #https://huggingface.co/bartowski/nvidia_OpenReasoning-Nemotron-14B-GGUF
                    ),
                )

                NEMOTRON_14B_Q4 = ModelSpec(
                    name="Nemotron-Qwen2.4-14B-Q4",
                    tag="nemotron_qwen_14B",
                    location=Location.LOCAL,
                    backend=Backend.LLAMACPP,
                    agent_backend=AgentBackend.LLAMACPPAGENT,
                    kind=ModelKind.HYBRID,
                    source=SourceHuggingface(
                        repo_id="bartowski/nvidia_OpenReasoning-Nemotron-14B-GGUF",
                        filename="nvidia_OpenReasoning-Nemotron-14B-Q4_0.gguf",
                        local_dir="data/model/"
                        #https://huggingface.co/bartowski/nvidia_OpenReasoning-Nemotron-14B-GGUF
                    ),
                )

                VANILLA_8B = ModelSpec(
                    name="Qwen3-8B",
                    tag="qwen3_8b",
                    location=Location.LOCAL,
                    backend=Backend.LLAMACPP,
                    agent_backend=AgentBackend.LLAMACPPAGENT,
                    kind=ModelKind.HYBRID,
                    source=SourceHuggingface(
                        repo_id="Qwen/Qwen3-8B-GGUF",
                        filename="Qwen3-8B-Q4_K_M.gguf",
                        local_dir="data/model/"
                        #https://huggingface.co/Qwen/Qwen3-8B-GGUF?show_file_info=Qwen3-8B-Q8_0.gguf
                    ),
                )

            class Llama3(Enum):
                LLAMA3_GROQ_8B_Q8 = ModelSpec(
                    name="Llama3-Groq-8B-Q8-Toolcalling",
                    tag="llama3_groq_8b_q8",
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

                LLAMA_3_1_8B_Q8 = ModelSpec(
                    name="Meta-Llama-3.1-8B-Instruct-GGUF",
                    tag="llama3.1_groq_8b_q8",
                    location=Location.LOCAL,
                    backend=Backend.LLAMACPP,
                    agent_backend=AgentBackend.LLAMACPPAGENT,
                    kind=ModelKind.HYBRID,
                    source=SourceHuggingface(
                        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
	                    filename="Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",

                        local_dir="data/model/"
                        #https://huggingface.co/rumbleFTW/Llama-3-Groq-8B-Tool-Use-Q8_0-GGUF
                    ),
                )

            GLM_9B = ModelSpec(
                name="GLM_9B-Q8",
                tag="glm_9b",
                location=Location.LOCAL,
                backend=Backend.LLAMACPP,
                agent_backend=AgentBackend.LLAMACPPAGENT,
                kind=ModelKind.HYBRID,
                source=SourceHuggingface(
                    repo_id="legraphista/glm-4-9b-chat-GGUF",
                    filename="glm-4-9b-chat.Q8_0.gguf",
                    local_dir="data/model/"
                    #https://huggingface.co/legraphista/glm-4-9b-chat-GGUF/blob/main/glm-4-9b-chat.Q8_0.gguf?library=llama-cpp-python
                ),
            )

            MPT_30B = ModelSpec(
                name="MPT-30B-Q2",
                tag="mpt_30b",
                location=Location.LOCAL,
                backend=Backend.LLAMACPP,
                agent_backend=AgentBackend.LLAMACPPAGENT,
                kind=ModelKind.HYBRID,
                source=SourceHuggingface(
                    repo_id="maddes8cht/mosaicml-mpt-30b-gguf",
                    filename="mosaicml-mpt-30b-Q2_K.gguf",
                    local_dir="data/model/"
                    #https://huggingface.co/maddes8cht/mosaicml-mpt-30b-gguf
                ),
            )

            PHI4_MINI_THINK_3_8B = ModelSpec(
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

            ARCH_AGENT_3B = ModelSpec(
                name="Arch-Agent-3B",
                tag="arch_agent_3b",
                location=Location.LOCAL,
                backend=Backend.LLAMACPP,
                agent_backend=AgentBackend.LLAMACPPAGENT,
                kind=ModelKind.TOOL,
                source=SourceHuggingface(
                    repo_id="Mungert/Arch-Agent-3B-GGUF",
	                filename="Arch-Agent-3B-bf16.gguf",
                    local_dir="data/model/"
                    #https://huggingface.co/Mungert/Arch-Agent-3B-GGUF
                ),
            )

            ARCH_AGENT_3B_Q6 = ModelSpec(
                name="Arch-Agent-3B-Q6",
                tag="arch_agent_3b_q6",
                location=Location.LOCAL,
                backend=Backend.LLAMACPP,
                agent_backend=AgentBackend.LLAMACPPAGENT,
                kind=ModelKind.TOOL,
                source=SourceHuggingface(
                    repo_id="Mungert/Arch-Agent-3B-GGUF",
	                filename="Arch-Agent-3B-q6_k_m.gguf",
                    local_dir="data/model/"
                    #https://huggingface.co/Mungert/Arch-Agent-3B-GGUF
                ),
            )

            XLAM2_8B = ModelSpec(
                name="XLAM2_8B",
                tag="xlam2-8B",
                location=Location.LOCAL,
                backend=Backend.LLAMACPP,
                agent_backend=AgentBackend.LLAMACPPAGENT,
                kind=ModelKind.TOOL,
                source=SourceHuggingface(
                    repo_id="Salesforce/xLAM-2-3b-fc-r-gguf",
	                filename="xLAM-2-3B-fc-r-Q8_0.gguf",
                    local_dir="data/model/"
                    #https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-r-gguf?show_file_info=Llama-xLAM-2-8B-fc-r-Q8_0.gguf&library=llama-cpp-python
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
                        #https://ollama.com/library/qwen3-coder
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
                    tag="llama3.1_nemotron_8b",
                    location=Location.LOCAL,
                    backend=Backend.OLLAMA,
                    agent_backend=AgentBackend.LANGCHAIN,
                    kind=ModelKind.HYBRID,
                    source=SourceOllama(
                        model_id="Randomblock1/nemotron-nano:8b",
                        #https://https://ollama.com/Randomblock1/nemotron-nano
                    ),
                )

            PHI4_MINI_3_8B = ModelSpec(
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

            GPT_OSS_20B = ModelSpec(
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

            class Cogito(Enum):
                COGITO_8B = ModelSpec(
                    name="Cogito-8B",
                    tag="cogito_8b",
                    location=Location.LOCAL,
                    backend=Backend.OLLAMA,
                    agent_backend=AgentBackend.LANGCHAIN,
                    kind=ModelKind.HYBRID,
                    source=SourceOllama(
                        model_id="cogito:8b",
                        #https://ollama.com/library/cogito
                    ),
                )

                COGITO_14B = ModelSpec(
                    name="Cogito-14B",
                    tag="cogito_14b",
                    location=Location.LOCAL,
                    backend=Backend.OLLAMA,
                    agent_backend=AgentBackend.LANGCHAIN,
                    kind=ModelKind.HYBRID,
                    source=SourceOllama(
                        model_id="cogito:8b",
                        #https://ollama.com/library/cogito
                    ),
                )

            class Exaone(Enum):
                EXAONE_32B = ModelSpec(
                    name="Exaone_32B",
                    tag="exaone_32b",
                    location=Location.LOCAL,
                    backend=Backend.OLLAMA,
                    agent_backend=AgentBackend.LANGCHAIN,
                    kind=ModelKind.HYBRID,
                    source=SourceOllama(
                        model_id="exaone-deep:32b",
                        #https://ollama.com/library/exaone-deep
                    ),
                )

            class Granite(Enum):
                GRANITE4_32B = ModelSpec(
                    name="Granite4-32B",
                    tag="granite4_32b",
                    location=Location.LOCAL,
                    backend=Backend.OLLAMA,
                    agent_backend=AgentBackend.LANGCHAIN,
                    kind=ModelKind.HYBRID,
                    source=SourceOllama(
                        model_id="granite4:32b-a9b-h",
                        #https://ollama.com/library/granite4/tags
                    ),
                )

                GRANITE4_7B = ModelSpec(
                    name="Granite4-7B",
                    tag="granite4_7b",
                    location=Location.LOCAL,
                    backend=Backend.OLLAMA,
                    agent_backend=AgentBackend.LANGCHAIN,
                    kind=ModelKind.HYBRID,
                    source=SourceOllama(
                        model_id="granite4:7b-a1b-h",
                        #https://ollama.com/library/granite4/tags
                    ),
                )

                GRANITE4_3B = ModelSpec(
                    name="Granite4-3B",
                    tag="granite4_3b",
                    location=Location.LOCAL,
                    backend=Backend.OLLAMA,
                    agent_backend=AgentBackend.LANGCHAIN,
                    kind=ModelKind.HYBRID,
                    source=SourceOllama(
                        model_id="granite4:3b",
                        #https://ollama.com/library/granite4/tags
                    ),
                )

                GRANITE4_1B = ModelSpec(
                    name="Granite4-1B",
                    tag="granite4_1b",
                    location=Location.LOCAL,
                    backend=Backend.OLLAMA,
                    agent_backend=AgentBackend.LANGCHAIN,
                    kind=ModelKind.HYBRID,
                    source=SourceOllama(
                        model_id="granite4:1b",
                        #https://ollama.com/library/granite4/tags
                    ),
                )

                GRANITE4_350M = ModelSpec(
                    name="Granite4-1B",
                    tag="granite4_350m",
                    location=Location.LOCAL,
                    backend=Backend.OLLAMA,
                    agent_backend=AgentBackend.LANGCHAIN,
                    kind=ModelKind.HYBRID,
                    source=SourceOllama(
                        model_id="granite4:350m",
                        #https://ollama.com/library/granite4/tags
                    ),
                )

            #UNFROTUNATLY NO TOOL SUPPORT

            #GLM_9B = ModelSpec(
            #    name="GLM-9B-Q8",
            #    tag="glm_9b",
            #    location=Location.LOCAL,
            #    backend=Backend.OLLAMA,
            #    agent_backend=AgentBackend.LANGCHAIN,
            #    kind=ModelKind.TOOL,
            #    source=SourceOllama(
            #        model_id="glm4:9b-chat-q8_0",
            #        #https://ollama.com/library/glm4/tags
            #    ),
            #)
#
            #XLAM2_8B_Q8 = ModelSpec(
            #    name="xLAM-2-8B-Q8",
            #    tag="xlam2_8b",
            #    location=Location.LOCAL,
            #    backend=Backend.OLLAMA,
            #    agent_backend=AgentBackend.LANGCHAIN,
            #    kind=ModelKind.TOOL,
            #    source=SourceOllama(
            #        model_id="robbiemu/Salesforce_Llama-xLAM-2:8b-fc-r-q8_0",
            #        #https://ollama.com/robbiemu/Salesforce_Llama-xLAM-2/tags
            #    ),
            #)
#
            #XLAM2_32B = ModelSpec(
            #    name="xLAM-2-32B",
            #    tag="xlam2_32b",
            #    location=Location.LOCAL,
            #    backend=Backend.OLLAMA,
            #    agent_backend=AgentBackend.LANGCHAIN,
            #    kind=ModelKind.TOOL,
            #    source=SourceOllama(
            #        model_id="felixnguyen95/xLAM-2-32b-fc-r:latest",
            #        #https://ollama.com/felixnguyen95/xLAM-2-32b-fc-r:latest
            #    ),
            #)

            MAGISTRAL_SMALL_24B = ModelSpec(
                name="Magistral-Small-24B-Q4",
                tag="magistral_small_24B",
                location=Location.LOCAL,
                backend=Backend.OLLAMA,
                agent_backend=AgentBackend.LANGCHAIN,
                kind=ModelKind.HYBRID,
                source=SourceOllama(
                    model_id="magistral:24b",
                    #https://ollama.com/library/magistral/tags
                ),
            )

            DEEPSEEK_R1_QWEN_14B = ModelSpec(
                name="Deepseek-R1-Qwen-Distill-14B",
                tag="deepseek_qwen_14B",
                location=Location.LOCAL,
                backend=Backend.OLLAMA,
                agent_backend=AgentBackend.LANGCHAIN,
                kind=ModelKind.HYBRID,
                source=SourceOllama(
                    model_id="deepseek-r1:14b-qwen-distill-q8_0",
                    #https://ollama.com/library/deepseek-r1/tags
                ),
            )

            DOLPHIN3_8B = ModelSpec(
                name="Dolphin3-8B",
                tag="dolphin3_8b",
                location=Location.LOCAL,
                backend=Backend.OLLAMA,
                agent_backend=AgentBackend.LANGCHAIN,
                kind=ModelKind.INSTRUCT,
                source=SourceOllama(
                    model_id="dolphin3:8b",
                    #https://ollama.com/library/dolphin3
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
                    endpoint_url="http://127.0.0.1:11434/v1",
                    model_id="gpt-oss:120b",
                ),
            )

            MAGISTRAL_SMALL_24B = ModelSpec(
                name="Magistral-Small-24B" ,
                tag="mistral_small_24b",
                location=Location.REMOTE,
                backend=Backend.OTHER,
                agent_backend=AgentBackend.LLAMACPPAGENT,
                kind=ModelKind.HYBRID,
                source=SourceRemote(
                    endpoint_url="http://127.0.0.1:11444/v1",
                    model_id="magistral:24b-small-2506-q8_0",
                ),
            )

            NEMOTRON_SUPER_49B = ModelSpec(
                name="Llama-3.3-Nemotron-Super-49B" ,
                tag="nemotron_super_49b",
                location=Location.REMOTE,
                backend=Backend.OTHER,
                agent_backend=AgentBackend.LLAMACPPAGENT,
                kind=ModelKind.HYBRID,
                source=SourceRemote(
                    endpoint_url="http://127.0.0.1:11444/v1",
                    model_id="MHKetbi/nvidia_Llama-3.3-Nemotron-Super-49B-v1:q8_0",
                ),
            )

            # institut
            MISTRAL_SMALL_24B = ModelSpec(
                name="Mistral-Small-24B" ,
                tag="mistral_small_24b",
                location=Location.REMOTE,
                backend=Backend.OTHER,
                agent_backend=AgentBackend.LLAMACPPAGENT,
                kind=ModelKind.HYBRID,
                source=SourceRemote(
                    endpoint_url="http://127.0.0.1:11444/v1",
                    model_id="hf.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M",
                ),
            )

            DEEPSEEK_R1_70B = ModelSpec(
                name="Deepseek-R1-70B" ,
                tag="deepseek_r1_70b",
                location=Location.REMOTE,
                backend=Backend.OTHER,
                agent_backend=AgentBackend.LANGCHAIN,
                kind=ModelKind.HYBRID,
                source=SourceRemote(
                    endpoint_url="http://127.0.0.1:11444/v1",
                    model_id="deepseek-r1:70b-llama-distill-fp16",
                ),
            )

            QWEN3 = ModelSpec(
                name="Qwen3" ,
                tag="qwen3",
                location=Location.REMOTE,
                backend=Backend.OTHER,
                agent_backend=AgentBackend.LANGCHAIN,
                kind=ModelKind.HYBRID,
                source=SourceRemote(
                    endpoint_url="http://127.0.0.1:11444/v1",
                    model_id="qwen3:235b-a22b-thinking-2507-q4_K_M",
                ),
            )

            # openai
            #GPT4 = ModelSpec(
            #    name="OpenAI-GPT-4",
            #    tag="gpt4",
            #    location=Location.REMOTE,
            #    backend=Backend.OTHER,
            #    agent_backend=AgentBackend.LANGCHAIN,
            #    kind=ModelKind.HYBRID,
            #    source=SourceRemote(
            #        endpoint_url="https://api.openai.com/v1",
            #        model_id="gpt-4",
            #    ),
            #)
#
            ## openai
            #GPT5 = ModelSpec(
            #    name="OpenAI-GPT-5",
            #    tag="gpt5",
            #    location=Location.REMOTE,
            #    backend=Backend.OTHER,
            #    agent_backend=AgentBackend.LANGCHAIN,
            #    kind=ModelKind.HYBRID,
            #    source=SourceRemote(
            #        endpoint_url="https://api.openai.com/v1",
            #        model_id="gpt-5",
            #    ),
            #)

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
