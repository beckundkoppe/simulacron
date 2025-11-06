# test_main.py



import game
from llm.cache import Cache
from llm.model import Model
    
def main():
    cache = Cache()
    #cache.get(Model.Local.LlamaCpp.HYBRID_PHI4_MINI_3_8B)
    #cache.get(Model.Local.Ollama.HYBRID_LLAMA3_1_NEMOTRON_8B)
    #cache.get(Model.Local.LlamaCpp.HYBRID_DEEPSEEK_QWEN_7B)
    #cache.get(Model.Local.Ollama.HYBRID_GPT_OSS_20B)

    #TEST#############################
    #game.test_run()
    #TEST#############################

    game.run(cache, Model.Local.LlamaCpp.HYBRID_DEEPSEEK_QWEN_7B)

if __name__ == "__main__":
    main()
