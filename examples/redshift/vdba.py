# Example
import ezsop
from ezsop import Event, Procedure, handler, End

def example():
    import enum
    from dataclasses import dataclass
    class tFindStatus(enum.Enum):
        ABORTED = enum.auto()
        FAILED = enum.auto()
        CANCELED = enum.auto()
        FINISHED = enum.auto()

    @dataclass
    class tStatusFound:
        status: tFindStatus
    class eStatusFound(Event[tStatusFound]): ...

    class eFindErrorMessage(Event[str]): ...

    class eFindExecutionTime(Event[int]): ...

    class FindStatus_1(Procedure[dict]):
        action = "Step 1: Find the status of the slow query using findStatus"
        ignore = []

        @handler
        def handle(self, state: dict, e: eStatusFound) -> 'str | Procedure[dict]':
            if e.payload.status == tFindStatus.ABORTED:
                return End("finish the investigation by saying the query was aborted.")
            elif e.payload.status == tFindStatus.FAILED:
                return ReportFailure_2()
            elif e.payload.status == tFindStatus.CANCELED:
                return End("finish the investigation by saying the query was canceled.")
            else:
                return InvestigteFinished()

    class ReportFailure_2(Procedure[dict]):
        action = "Report failure"
        ignore = []

        @handler
        def handle(self, state: dict, e: eFindErrorMessage) -> 'str | Procedure[dict]':
            return End(f"The query failed with error: {e.payload}")


    class InvestigteFinished(Procedure[dict]):
        action = "the status is finished, that means query ran successfully and we need to debug why it is slow."
        "Run the action findExecutionTime to first verify if query is indeed slow."
        ignore = []

        @handler
        def handle(self, state: dict, e: eFindExecutionTime) -> 'str | Procedure[dict]':
            if e.payload < 60:
                return End(f"The query is not slow, execution time: {e.payload} units.")
            else:
                return End(f"The query is slow, execution time: {e.payload} units.")
    
    class FetchQueryPlan(Procedure[dict]):
        action = "Fetch the query plan to understand why the query is slow."
        ignore = []

        @handler
        def handle(self, state: dict, e: Event[str]) -> 'str | Procedure[dict]':
            return End(f"Query plan fetched: {e.payload}")