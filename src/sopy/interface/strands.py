# support of strands agents events
from strands.types.tools import ToolResult
from ezsop.core.event import Event, EventTranslator


class eStrandsToolMsg(Event[ToolResult]): ...

@EventTranslator.register()
def translate_strands_tool_msg(event: ToolResult) -> eStrandsToolMsg:
    """
    Translate a ToolResult to an eStrandsToolMsg event.
    """
    return eStrandsToolMsg(payload=event, description=f"Strands Tool Message ({event['toolUseId']})")