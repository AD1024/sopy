from typing import Tuple, List
from sopy import Event

class eReadDirectory(Event[Tuple[str, List[str] | None]]): ...

class eFileDeleteRequest(Event[str]): # ToolUse message
    def __init__(self, toolUseId: str, path: str):
        super().__init__(toolUseId=toolUseId, payload=path)

class eFileDeleted(Event[str | None]): ... # ToolMessage

class eCopyRequest(Event[Tuple[str, str]]):
    def __init__(self, toolUseId: str, source: str, dest: str):
        super().__init__(toolUseId=toolUseId, payload=(source, dest))

class eFileCopied(Event[Tuple[str, str] | None]): ...
