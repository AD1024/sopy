from typing import Optional, Any, Generic, TypeVar, Callable


T = TypeVar('T', bound=Any)
class Event(Generic[T]):
    """
    The base class of an event
    """
    description: Optional[str]
    sender: Any
    target: Any
    payload: T

    def __init__(self, payload: T, description: Optional[str] = None, sender: Any = None, target: Any = None):
        """
        Initialize the event with a payload, description, sender, and target.
        
        Args:
            payload: The data associated with the event.
            description: Optional description of the event.
            sender: The entity that sent the event.
            target: The entity that is the intended recipient of the event.
        """
        self.payload = payload
        self.description = description
        self.sender = sender
        self.target = target

class EventTranslator:

    translator_map: dict[type, Callable[[Any], Event[Any]]]

    @staticmethod
    def register():
        """Decorator to register a function as an event translator."""
        def decorator(func: Callable[[Any], Event[Any]]) -> Callable[[Any], Event[Any]]:
            assert len(func.__annotations__) == 1, "Translator function must have exactly one type annotation."
            event_type = next(iter(func.__annotations__.values()))
            EventTranslator.translator_map[event_type] = func
            return func
        return decorator
    
    @staticmethod
    def translate(event: Any) -> Optional[Event[Any]]:
        translator = EventTranslator.translator_map.get(type(event))
        if translator:
            return translator(event)
        return None


def show(event: Event) -> str:
    """
    Show an event
    """
    return f"{event.__class__.__name__}(description={event.description}, sender={event.sender}, target={event.target}, payload={event.payload})"