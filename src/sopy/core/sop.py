from copy import deepcopy
import uuid
import traceback
from typing import Generic, Iterable, Any, Callable, Self, Type
from .event import Event, EventBus
from .procedure import Sigma, Procedure, End
from ..prompt import Prompt, make_prompt
from ..utils import Log


class SOPViolation(Exception):
    """
    Exception raised when a SOP violation occurs.
    This is used to signal that the SOP has encountered an unexpected event or state.
    """
    def __init__(self, message: str):
        super().__init__(message)

class SOPRetryExceptionHook(Exception):
    """
    Exception raised when a SOP retry exception hook is triggered.
    This is used to signal that the SOP has encountered an error that can be retried.
    """
    def __init__(self, message: Prompt, handler: Callable[[Prompt], Any], current_proceure: Procedure):
        super().__init__(message.prompt)
        self.prompt = message
        self.handler = handler
        self.current_procedure = current_proceure
    
    def execute(self):
        return self.handler(self.prompt)

class SOP(Generic[Sigma]):

    def __init__(self, state: Sigma, init_proc: Procedure, error_handler: Callable[[Prompt], Any]=lambda *x: x, ignore_unrecongnized_events: bool = True, retry_limits: int = 3):
        """
        Initialize the SOP with the given state.
        """
        self._sop_ctx = False
        self.state_backup = deepcopy(state)
        self.state = deepcopy(state)
        self.init_proc = deepcopy(init_proc)
        self.current_proc = deepcopy(init_proc)
        self.ignore_unrecongnized_events = ignore_unrecongnized_events
        self.error_handler = error_handler
        self.obs_events = set()
        self.obs_agents = set()
        self.retry_limits = retry_limits
        self.num_retry = 0
        self._id = uuid.uuid4().hex
    
    def observes(self, *args: Type[Event]) -> Self:
        for e in args:
            if not issubclass(e, Event):
                raise TypeError(f"Expected Event type, got {e}.")
            self.obs_events.add(e)
        return self

    def from_agent(self, *agents: Any) -> Self:
        self.obs_agents = self.obs_agents.union(agents)
        return self
    
    def with_recovery(self, error_handler: Callable[[Prompt], Any], retry_limits: int = 3) -> Self:
        """
        Set the error handler and retry limits for the SOP.
        This allows the SOP to recover from errors by retrying the operation.
        
        Args:
            retry_limits (int): The number of times to retry the SOP before giving up.
            error_handler (Callable[[Prompt], Any]): The function to call when an error occurs.
        """
        self.error_handler = error_handler
        self.retry_limits = retry_limits
        return self

    def try_recover(self, prompt: Prompt):
        if self.num_retry == self.retry_limits:
            Log.error("Retry limits exceeded. SOP cannot continue.")
            exit(1)
        self.num_retry += 1
        raise SOPRetryExceptionHook(prompt, self.error_handler, self.current_proc)

    def __and__(self, other: 'SOP') -> 'SOP':
        if not isinstance(other, SOP):
            raise TypeError(f"Expected SOP type, got {type(other)}.")
        return Conjunction(self, other)

    def __call__(self, e: Event):
        if len(self.obs_events) and not isinstance(e, tuple(self.obs_events)):
            return
        if len(self.obs_agents):
            if e.sender not in self.obs_agents and e.target not in self.obs_agents:
                return
        if not self._sop_ctx:
            raise RuntimeError(f"`observe` should not be called outside of SOP context. Use `with` to enter the context.")
        try:
            Log.debug(f"{self.__class__.__name__} Now at " + str(self.current_proc) + f" handling {e}")
            nxt = self.current_proc._handle(self.state, e)
            if isinstance(nxt, str):
                self.try_recover(make_prompt(nxt))
            elif isinstance(nxt, Procedure):
                self.current_proc = nxt
            else:
                raise ValueError(f"Unexpected return type from handler: {type(nxt)}. Expected str or Procedure.")
        except (AssertionError, SOPViolation) as ex:
            self.try_recover(make_prompt(f"SOP Violation detected: {ex}. Please try again."))
        except SOPRetryExceptionHook as ex:
            raise ex
        except Exception as ex:
            # usually internal errors due to bugs in SOP implementation
            Log.error(f"Internal error in SOP: {ex}. Please check the implementation.")
            traceback.print_exc()
            exit(1)

    def __enter__(self):
        """
        Enter the SOP context, setting the SOP context flag.
        """
        self._sop_ctx = True
        self.num_retry = 0
        EventBus.subscribe(self._id, self)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the SOP context, resetting the SOP context flag.
        """
        if exc_type is not None:
            # If an exception occurred, we can handle it here if needed
            Log.error(f"Exception occurred: {exc_value}. SOP context exited with error.")
            return False
        if not isinstance(self.current_proc, End):
            prompt_text = f"SOP left unfinished: current procedure is {self.current_proc.__class__.__name__} with prompt {self.current_proc.prompt}."
            if self.num_retry < self.retry_limits:
                self.error_handler(make_prompt(prompt_text))
                self.num_retry += 1
                return self.__exit__(exc_type, exc_value, traceback)
            else:
                raise SOPViolation(prompt_text)
        self._sop_ctx = False
        EventBus.unsubscribe(self._id)
        self.state = deepcopy(self.state_backup)
        self.current_proc = deepcopy(self.init_proc)
        return True

class Conjunction(SOP[Sigma]):
    def __init__(self, *sops: SOP):
        self.conjoined = list(sops)

    def __and__(self, other: SOP) -> Self:
        if not isinstance(other, SOP):
            raise TypeError(f"Expected SOP type, got {type(other)}.")
        self.conjoined.append(other)
        return self

    def __enter__(self):
        for sop in self.conjoined:
            sop.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        for sop in self.conjoined:
            if not sop.__exit__(exc_type, exc_value, traceback):
                Log.error(f"Conjunction SOP exited with error in {sop.__class__.__name__}.")
                return False
        return super().__exit__(exc_type, exc_value, traceback)
