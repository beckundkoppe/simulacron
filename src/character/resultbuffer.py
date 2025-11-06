

from typing import List

class Result:
    pass

class FormalError(Result):
    def __init__(self, what: str):
        self.what = what
        Resultbuffer.buffer.append(self)

class ActionNotPossible(Result):
    def __init__(self, what: str):
        self.what = what
        Resultbuffer.buffer.append(self)

class Success(Result):
    def __init__(self, what: str):
        self.what = what
        Resultbuffer.buffer.append(self)

class Resultbuffer:
    buffer: List[Result] = []