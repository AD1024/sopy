from copy import deepcopy
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

    def __init__(self, state: Sigma, init_proc: Procedure, ignore_unrecongnized_events: bool = True):
        """
        Initialize the SOP with the given state.
        """
        self._sop_ctx = False
        self.state_backup = deepcopy(state)
        self.state = deepcopy(state)
        self.init_proc = deepcopy(init_proc)
        self.current_proc = deepcopy(init_proc)
        self.ignore_unrecongnized_events = ignore_unrecongnized_events

    def unwrap_iter(self, stream: Iterable) -> Iterable[Event[Any]]:
        # TODO: implement for LangGraph and strands
        ...

    def observe(self, stream: Iterable) -> Iterable:
        if not self._sop_ctx:
            raise RuntimeError(f"`observe` should not be called outside of SOP context. Use `with` to enter the context.")
        for e in self.unwrap_iter(stream):
            try:
                nxt = self.current_proc._handle(self.state, e)
                if isinstance(nxt, str):
                    yield make_prompt(nxt)
                elif isinstance(nxt, Procedure):
                    self.current_proc = nxt
                    yield e
                else:
                    raise ValueError(f"Unexpected return type from handler: {type(nxt)}. Expected str or Procedure.")
            except AssertionError | SOPViolation as ex:
                yield make_prompt(f"SOP Violation detected: {ex}. Please try again.")
            except Exception as ex:
                # usually internal errors due to bugs in SOP implementation
                raise ex
        if not isinstance(self.current_proc, End):
            yield make_prompt(f"SOP left unfinished: current procedure is {self.current_proc.__class__.__name__}.")

    def __enter__(self):
        """
        Enter the SOP context, setting the SOP context flag.
        """
        self._sop_ctx = True
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the SOP context, resetting the SOP context flag.
        """
        self._sop_ctx = False
        self.state = deepcopy(self.state_backup)
        self.current_proc = deepcopy(self.init_proc)
        if exc_type is not None:
            # If an exception occurred, we can handle it here if needed
            print(f"Exception occurred: {exc_value}. SOP context exited with error.")
            return False
        else:
            return True