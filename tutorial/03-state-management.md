# State Management: The Sigma Pattern in SOPy

State management is crucial for building reliable agent workflows. SOPy uses the "Sigma pattern" where all procedures operate on a shared state object, enabling coordinated multi-step operations. This tutorial explains how to design and manage state effectively in SOPy.

## Understanding the Sigma Pattern

In SOPy, `Sigma` (Σ) represents the shared state that flows through your entire workflow:

```python
from typing import TypeVar
from sopy import Procedure

# Define your state type
Sigma = TypeVar('Sigma')

class MyProcedure(Procedure[Sigma]):
    @handler
    def handle_event(self, sigma: Sigma, event: MyEvent):
        # sigma is the shared state object
        # All procedures operate on the same sigma instance
        sigma.some_field = event.payload
        return NextProcedure()
```

This pattern provides:
- **Shared context** across all procedures
- **Type safety** through generic typing
- **State evolution** as the workflow progresses
- **Consistency** through centralized state management

## Designing State Objects

### 1. Using Pydantic Models (Recommended)

Pydantic provides excellent state management with validation and type safety:

```python
from pydantic import BaseModel, Field
from typing import List, Set, Optional
from datetime import datetime

class WorkflowState(BaseModel):
    # Basic fields
    workflow_id: str = Field(..., description="Unique identifier for this workflow")
    started_at: datetime = Field(default_factory=datetime.now, description="When the workflow started")
    
    # Progress tracking
    current_step: str = Field(default="", description="Current step in the workflow")
    completed_steps: List[str] = Field(default_factory=list, description="Steps that have been completed")
    
    # Data accumulation
    collected_data: List[dict] = Field(default_factory=list, description="Data collected during workflow")
    processed_results: Optional[dict] = Field(default=None, description="Final processed results")
    
    # Error tracking
    error_count: int = Field(default=0, description="Number of errors encountered")
    last_error: Optional[str] = Field(default=None, description="Most recent error message")
    
    # Configuration
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    timeout_seconds: int = Field(default=300, description="Workflow timeout in seconds")
```

### 2. Rich State with Computed Properties

Add methods to your state for computed properties and validation:

```python
class TaskState(BaseModel):
    tasks: List[dict] = Field(default_factory=list)
    completed_tasks: Set[str] = Field(default_factory=set)
    failed_tasks: Set[str] = Field(default_factory=set)
    
    @property
    def total_tasks(self) -> int:
        return len(self.tasks)
    
    @property
    def completion_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return len(self.completed_tasks) / self.total_tasks
    
    @property
    def is_complete(self) -> bool:
        return len(self.completed_tasks) == self.total_tasks
    
    @property
    def has_failures(self) -> bool:
        return len(self.failed_tasks) > 0
    
    def add_task(self, task_id: str, task_data: dict):
        """Add a new task to the workflow"""
        self.tasks.append({"id": task_id, **task_data})
    
    def mark_completed(self, task_id: str):
        """Mark a task as completed"""
        self.completed_tasks.add(task_id)
        self.failed_tasks.discard(task_id)  # Remove from failed if present
    
    def mark_failed(self, task_id: str):
        """Mark a task as failed"""
        self.failed_tasks.add(task_id)
        self.completed_tasks.discard(task_id)  # Remove from completed if present
```

### 3. Nested State Structures

For complex workflows, organize state hierarchically:

```python
class DatabaseConfig(BaseModel):
    host: str
    port: int
    database: str
    username: str
    connection_pool_size: int = 10

class ProcessingConfig(BaseModel):
    batch_size: int = 100
    parallel_workers: int = 4
    timeout_per_batch: int = 60

class ETLState(BaseModel):
    # Configuration
    db_config: DatabaseConfig
    processing_config: ProcessingConfig
    
    # Runtime state
    current_batch: int = 0
    total_batches: int = 0
    processed_records: int = 0
    
    # Data tracking
    extraction_results: List[dict] = Field(default_factory=list)
    transformation_errors: List[dict] = Field(default_factory=list)
    load_statistics: dict = Field(default_factory=dict)
    
    @property
    def progress_percentage(self) -> float:
        if self.total_batches == 0:
            return 0.0
        return (self.current_batch / self.total_batches) * 100
```

## State Evolution Patterns

### 1. Progressive Data Accumulation

State grows as the workflow progresses:

```python
class DataCollectionState(BaseModel):
    sources: List[str] = Field(default_factory=list)
    raw_data: dict = Field(default_factory=dict)
    cleaned_data: dict = Field(default_factory=dict)
    analyzed_data: dict = Field(default_factory=dict)
    final_report: Optional[dict] = None

class CollectData(Procedure[DataCollectionState]):
    @handler
    def handle_data_source_read(self, sigma: DataCollectionState, event: DataSourceRead):
        source_name = event.payload["source"]
        data = event.payload["data"]
        
        # Accumulate data from multiple sources
        sigma.raw_data[source_name] = data
        sigma.sources.append(source_name)
        
        # Check if all sources collected
        if len(sigma.sources) >= sigma.expected_sources:
            return CleanData()
        
        return self  # Continue collecting

class CleanData(Procedure[DataCollectionState]):
    @handler
    def handle_cleaning_complete(self, sigma: DataCollectionState, event: CleaningComplete):
        sigma.cleaned_data = event.payload["cleaned_data"]
        return AnalyzeData()
```

### 2. State Validation and Constraints

Ensure state consistency through validation:

```python
class OrderState(BaseModel):
    order_id: str
    items: List[dict] = Field(default_factory=list)
    total_amount: float = 0.0
    payment_status: str = "pending"
    shipping_address: Optional[dict] = None
    
    @validator('payment_status')
    def validate_payment_status(cls, v):
        valid_statuses = ["pending", "processing", "completed", "failed"]
        if v not in valid_statuses:
            raise ValueError(f"Invalid payment status: {v}")
        return v
    
    @validator('total_amount')
    def validate_total_amount(cls, v):
        if v < 0:
            raise ValueError("Total amount cannot be negative")
        return v
    
    def add_item(self, item: dict):
        """Add item and update total"""
        self.items.append(item)
        self.total_amount += item.get("price", 0.0)
    
    def can_process_payment(self) -> bool:
        """Check if order is ready for payment processing"""
        return (
            len(self.items) > 0 and
            self.total_amount > 0 and
            self.shipping_address is not None and
            self.payment_status == "pending"
        )
```

### 3. State Checkpoints and Rollback

Implement checkpointing for error recovery:

```python
from copy import deepcopy

class CheckpointState(BaseModel):
    # Main state
    current_data: dict = Field(default_factory=dict)
    processing_stage: str = "initial"
    
    # Checkpoint management
    checkpoints: dict = Field(default_factory=dict)
    
    def create_checkpoint(self, name: str):
        """Create a checkpoint of current state"""
        self.checkpoints[name] = {
            "data": deepcopy(self.current_data),
            "stage": self.processing_stage,
            "timestamp": datetime.now()
        }
    
    def restore_checkpoint(self, name: str) -> bool:
        """Restore state from checkpoint"""
        if name not in self.checkpoints:
            return False
        
        checkpoint = self.checkpoints[name]
        self.current_data = deepcopy(checkpoint["data"])
        self.processing_stage = checkpoint["stage"]
        return True
    
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints"""
        return list(self.checkpoints.keys())

class TransactionalProcedure(Procedure[CheckpointState]):
    @handler
    def handle_begin_operation(self, sigma: CheckpointState, event: BeginOperation):
        # Create checkpoint before risky operation
        sigma.create_checkpoint("before_operation")
        return ProcessOperation()
    
    @handler
    def handle_operation_failed(self, sigma: CheckpointState, event: OperationFailed):
        # Restore from checkpoint on failure
        if sigma.restore_checkpoint("before_operation"):
            return "Operation failed. State restored to checkpoint. Please try again."
        else:
            return "Operation failed and checkpoint restoration failed. Manual intervention required."
```

## State Access Patterns

### 1. Read-Only State Access

Some procedures only read state without modifying it:

```python
class ReportGenerator(Procedure[AnalyticsState]):
    @handler
    def handle_generate_report(self, sigma: AnalyticsState, event: GenerateReport):
        # Read state to generate report
        report = {
            "total_records": len(sigma.processed_data),
            "success_rate": sigma.success_count / sigma.total_count if sigma.total_count > 0 else 0,
            "error_summary": sigma.error_summary,
            "processing_time": sigma.end_time - sigma.start_time if sigma.end_time else None
        }
        
        # Don't modify sigma, just use it for reporting
        return SaveReport(report_data=report)
```

### 2. Conditional State Updates

Update state based on conditions:

```python
class ConditionalProcessor(Procedure[ProcessingState]):
    @handler
    def handle_processing_result(self, sigma: ProcessingState, event: ProcessingResult):
        result = event.payload
        
        if result["success"]:
            sigma.success_count += 1
            sigma.processed_items.append(result["item_id"])
            
            # Conditional logic based on state
            if sigma.success_count >= sigma.target_count:
                sigma.status = "completed"
                return CompletionProcedure()
            else:
                return self  # Continue processing
        else:
            sigma.error_count += 1
            sigma.failed_items.append(result["item_id"])
            
            # Check error threshold
            if sigma.error_count > sigma.max_errors:
                sigma.status = "failed"
                return ErrorHandlingProcedure()
            else:
                return "Processing failed for item. Continuing with next item."
```

### 3. State Aggregation

Combine data from multiple sources:

```python
class AggregationState(BaseModel):
    data_sources: dict = Field(default_factory=dict)
    aggregated_results: dict = Field(default_factory=dict)
    
    def add_source_data(self, source_name: str, data: dict):
        """Add data from a source"""
        self.data_sources[source_name] = data
    
    def compute_aggregates(self):
        """Compute aggregated statistics"""
        all_values = []
        for source_data in self.data_sources.values():
            all_values.extend(source_data.get("values", []))
        
        if all_values:
            self.aggregated_results = {
                "count": len(all_values),
                "sum": sum(all_values),
                "average": sum(all_values) / len(all_values),
                "min": min(all_values),
                "max": max(all_values)
            }

class AggregationProcedure(Procedure[AggregationState]):
    @handler
    def handle_source_data(self, sigma: AggregationState, event: SourceDataReceived):
        source = event.payload["source"]
        data = event.payload["data"]
        
        sigma.add_source_data(source, data)
        
        # Check if all sources received
        if len(sigma.data_sources) >= sigma.expected_sources:
            sigma.compute_aggregates()
            return GenerateReport()
        
        return self
```

## File System Example State Management

The file system operator demonstrates sophisticated state management:

```python
class FSInfo(BaseModel):
    # Path configuration
    src_path: str = Field(..., description="The source path from which files are to be migrated.")
    dst_path: str = Field(..., description="The destination path to which files are to be migrated.")
    backup_path: str = Field(..., description="The backup path where files are temporarily stored during migration.")
    
    # Progress tracking sets
    path_read: set[str] = Field(default_factory=set, description="Set of paths that have been read from the source directory.")
    path_copied: set[str] = Field(default_factory=set, description="Set of paths that have been copied to the destination directory.")
    path_backuped: set[str] = Field(default_factory=set, description="Set of paths that have been backed up to the backup directory.")
    
    # File tracking for cleanup
    files_in_backup: set[str] = Field(default_factory=set, description="Set of files that have been backed up to the backup directory.")
    files_in_migration: set[str] = Field(default_factory=set, description="Set of files that have been migrated to the destination directory.")
```

Key patterns demonstrated:
- **Descriptive fields**: Field descriptions are used in automatic prompt generation
- **Set-based tracking**: Using sets for efficient membership testing and progress tracking
- **Multi-phase state**: Different sets track different phases of the operation
- **Cleanup coordination**: State enables coordinated cleanup across procedures

## Best Practices

### 1. Design for Observability

Make state easy to inspect and debug:

```python
class ObservableState(BaseModel):
    # Add metadata for debugging
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    update_count: int = 0
    
    def update_timestamp(self):
        """Update the last modified timestamp"""
        self.last_updated = datetime.now()
        self.update_count += 1
    
    def get_debug_info(self) -> dict:
        """Get debugging information"""
        return {
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "update_count": self.update_count,
            "state_size": len(self.dict())
        }
```

### 2. Use Type Hints Consistently

Ensure type safety throughout your state:

```python
from typing import Dict, List, Optional, Union

class TypedState(BaseModel):
    # Use specific types
    user_ids: List[int] = Field(default_factory=list)
    user_data: Dict[int, dict] = Field(default_factory=dict)
    current_user: Optional[int] = None
    
    # Use unions for multiple valid types
    result: Union[dict, str, None] = None
    
    # Use custom types for complex data
    processing_status: Literal["pending", "processing", "completed", "failed"] = "pending"
```

### 3. Implement State Validation

Add validation to catch errors early:

```python
class ValidatedState(BaseModel):
    items: List[dict] = Field(default_factory=list)
    max_items: int = 100
    
    @validator('items')
    def validate_items_count(cls, v, values):
        max_items = values.get('max_items', 100)
        if len(v) > max_items:
            raise ValueError(f"Too many items: {len(v)} > {max_items}")
        return v
    
    @root_validator
    def validate_consistency(cls, values):
        """Validate overall state consistency"""
        items = values.get('items', [])
        if items:
            # Ensure all items have required fields
            for item in items:
                if 'id' not in item:
                    raise ValueError("All items must have an 'id' field")
        return values
```

### 4. Document State Evolution

Document how state changes through the workflow:

```python
class DocumentedState(BaseModel):
    """
    State for the data processing workflow.
    
    State Evolution:
    1. Initial: empty state with configuration
    2. DataCollection: raw_data populated
    3. DataCleaning: cleaned_data populated
    4. DataAnalysis: analysis_results populated
    5. ReportGeneration: final_report populated
    """
    
    # Configuration (set once)
    config: dict = Field(default_factory=dict, description="Workflow configuration")
    
    # Stage 1: Data Collection
    raw_data: dict = Field(default_factory=dict, description="Raw data from sources")
    
    # Stage 2: Data Cleaning
    cleaned_data: dict = Field(default_factory=dict, description="Cleaned and validated data")
    
    # Stage 3: Data Analysis
    analysis_results: dict = Field(default_factory=dict, description="Analysis results and statistics")
    
    # Stage 4: Report Generation
    final_report: Optional[dict] = Field(default=None, description="Final generated report")
```

## Summary

Effective state management in SOPy requires:

- **Clear state design** using Pydantic models with descriptive fields
- **Type safety** through consistent use of type hints
- **State evolution** patterns that accumulate data progressively
- **Validation** to ensure state consistency
- **Observability** features for debugging and monitoring
- **Documentation** of how state changes through the workflow

The Sigma pattern provides a powerful foundation for building complex, stateful agent workflows that can coordinate multiple procedures while maintaining consistency and type safety.

## Next Steps

- [Handlers and Transitions](04-handlers-and-transitions.md) - Learn about state transitions
- [Automatic Prompts](05-automatic-prompts.md) - See how state descriptions improve prompts
- [File System Example](08-file-system-example.md) - Complex state management in action
