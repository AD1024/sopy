import importlib
from typing import List, Optional, Callable, Generic, TypeVar, Any
from .event import Event
from ..prompt import Prompt, make_prompt
from functools import lru_cache


# Global state
Sigma = TypeVar('Sigma')
T = TypeVar('T')
T1 = TypeVar('T1')


def handler(func: Callable) -> Callable[['Procedure[Sigma]', Sigma, Event[Any]], 'str | Procedure[Sigma]']:
    setattr(func, 'is_handler', True)
    for _, v in func.__annotations__.items():
        if issubclass(v, Event):
            if hasattr(func, 'event_type'):
                raise ValueError(f"Handler {func.__name__} already has an event type set: {getattr(func, 'event_type')}.")
            setattr(func, 'event_type', v)
    return func


class Procedure(Generic[Sigma]):
    """
    Base class for a procedure in the SOP
    """

    prompt: Prompt = make_prompt("")
    ignore: List[Event[Any]] = []

    def recover_prompt(self, state: Sigma, event: Event[Any], exception: Optional[Exception]) -> str:
        """
        Generate a prompt for recovering from an unexpected event.
        If the event is expected, return None.
        """
        if exception:
            return f"An error occurred while handling event {event}: {exception}. Please try again."
        return f"Event {event} should not happen in procedure {self.__class__.__name__}. Please try again."

    def _get_handler(self, event_type) -> Optional[Callable[['Procedure[Sigma]', Sigma, Event[Any]], 'str | Procedure[Sigma]']]:
        """
        Get the handler for the event type.
        If no handler is found, return None.
        """
        for attr in dir(self):
            method: Optional[Callable[['Procedure[Sigma]', Sigma, Event[Any]], 'str | Procedure[Sigma]']] = getattr(self, attr)
            if callable(method) and hasattr(method, 'is_handler') and getattr(method, 'event_type', None) == event_type:
                return method
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.prompt})"
    
    def __str__(self) -> str:
        return self.__repr__()

    @lru_cache()
    def _is_class_method(self, handler):
        if hasattr(handler, '__self__'):
            return handler.__self__ is not None
        return False
    
    def _handle(self, state: Sigma, event: Event[Any]) -> 'str | Procedure[Sigma]':
        """
        Handle an event. If the event is unexpected, return a string prompt for recovery.
        If the event is expected, return the next procedure to execute.
        """
        if event in self.ignore:
            return self
        handler = self._get_handler(event.__class__)
        if handler:
            try:
                if not self._is_class_method(handler):
                    return handler(self, state, event)
                return handler(state, event)
            except Exception as e:
                return self.recover_prompt(state, event, e)
        else:
            return self.recover_prompt(state, event, None)
            

class End(Procedure[Sigma]):
    prompt: Prompt = make_prompt("Finish the procedure, and report related results to the user")

    def __init__(self, prompt: Prompt | None = None):
        self.prompt = prompt if prompt else self.prompt
