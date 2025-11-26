AGENT = None

ENTITY = None

EXTRA_MODEL = None


RESULT = None

ANY_ACTION = False


ANSWER_BUFFER: bool = None

ANSWER_BUFFER_REASON: str = None

def get_answer() -> bool:
    ans = ANSWER_BUFFER
    ANSWER_BUFFER = None
    return ans

def get_rationale() -> str:
    ans = ANSWER_BUFFER_REASON
    ANSWER_BUFFER_REASON = None
    return ans

def any_action() -> bool:
    ans = ANY_ACTION
    ANY_ACTION = False
    return ans