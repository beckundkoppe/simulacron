AGENT = None

ENTITY = None

RESULT = None

EXTRA_MODEL = None





ANSWER_BUFFER: bool = None

def get_answer() -> bool:
    ans = ANSWER_BUFFER
    ANSWER_BUFFER = None
    return ans