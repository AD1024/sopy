from typing import Generic, Iterable, Any
from .event import Event
from .procedure import Sigma, Procedure, End
from ..prompt import Prompt, make_prompt

class SOPViolation(Exception):
    """
    Exception raised when a SOP violation occurs.
    This is used to signal that the SOP has encountered an unexpected event or state.
    """
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

class SOP(Generic[Sigma]):

    def __init__(self, state: Sigma, init_proc: Procedure[Sigma], ignore_unrecongnized_events: bool = True):
        """
        Initialize the SOP with the given state.
        """
        self.state = state
        self.current_proc = init_proc
        self.ignore_unrecongnized_events = ignore_unrecongnized_events

    def unwrap_iter(self, stream: Iterable) -> Iterable[Event[Any]]:
        # TODO: implement for LangGraph and strands
        ...

    def observe(self, stream: Iterable) -> Prompt | None:
        for e in self.unwrap_iter(stream):
            try:
                nxt = self.current_proc._handle(self.state, e)
            except AssertionError | SOPViolation as ex:
                return make_prompt(f"SOP Violation detected: {ex}. Please try again.")
            except Exception as ex:
                # usually internal errors due to bugs in SOP implementation
                raise ex
            if isinstance(nxt, str):
                return make_prompt(nxt)
            elif isinstance(nxt, Procedure):
                self.current_proc = nxt
            else:
                raise ValueError(f"Unexpected return type from handler: {type(nxt)}. Expected str or Procedure.")
        if not isinstance(self.current_proc, End):
            return make_prompt(f"SOP left unfinished: current procedure is {self.current_proc.__class__.__name__}.")