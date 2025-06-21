# Advanced Patterns: Complex SOPy Workflows

This tutorial covers advanced patterns and techniques for building sophisticated SOPy workflows. These patterns address complex real-world scenarios that go beyond basic state machines.

## Multi-Agent Coordination

SOPy can coordinate multiple agents working together on complex tasks.

### 1. Agent Delegation Pattern

```python
from sopy.interface.strands import StrandsAgent

class TaskCoordinator(Procedure[CoordinationState]):
    def __init__(self):
        # Specialized agents for different tasks
        self.data_agent = StrandsAgent(
            tools=[query_database, process_data],
            system_prompt="You are a data processing specialist."
        )
        self.report_agent = StrandsAgent(
            tools=[generate_chart, create_document],
            system_prompt="You are a report generation specialist."
        )
        self.review_agent = StrandsAgent(
            tools=[validate_output, check_quality],
            system_prompt="You are a quality assurance specialist."
        )
    
    @handler
    def handle_task_assignment(self, state: CoordinationState, event: TaskAssignment):
        task = event.payload
        
        if task.type == "data_processing":
            # Delegate to data specialist
            result = self.data_agent(f"Process this data: {task.data}")
            return DataProcessingComplete(result=result)
        elif task.type == "report_generation":
            # Delegate to report specialist
            result = self.report_agent(f"Generate report from: {task.data}")
            return ReportGenerationComplete(result=result)
        elif task.type == "quality_review":
            # Delegate to review specialist
            result = self.review_agent(f"Review this output: {task.output}")
            return QualityReviewComplete(result=result)
        else:
            return f"Unknown task type: {task.type}"
```

### 2. Parallel Agent Execution

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelAgentCoordinator(Procedure[ParallelState]):
    def __init__(self):
        self.agents = {
            "agent_1": StrandsAgent(tools=[tool_set_1], system_prompt="Agent 1 prompt"),
            "agent_2": StrandsAgent(tools=[tool_set_2], system_prompt="Agent 2 prompt"),
            "agent_3": StrandsAgent(tools=[tool_set_3], system_prompt="Agent 3 prompt"),
        }
        self.executor = ThreadPoolExecutor(max_workers=3)
    
    @handler
    def handle_parallel_task_start(self, state: ParallelState, event: ParallelTaskStart):
        tasks = event.payload.tasks
        
        # Submit tasks to different agents in parallel
        futures = {}
        for task in tasks:
            agent_id = self._select_agent_for_task(task)
            future = self.executor.submit(
                self.agents[agent_id], 
                f"Execute task: {task.description}"
            )
            futures[task.id] = (agent_id, future)
        
        state.pending_tasks = futures
        return WaitForParallelCompletion()
    
    def _select_agent_for_task(self, task):
        """Select the best agent for a given task"""
        if task.category == "data":
            return "agent_1"
        elif task.category == "analysis":
            return "agent_2"
        else:
            return "agent_3"

class WaitForParallelCompletion(Procedure[ParallelState]):
    @handler
    def handle_check_completion(self, state: ParallelState, event: CheckCompletion):
        completed_tasks = {}
        remaining_tasks = {}
        
        # Check which tasks have completed
        for task_id, (agent_id, future) in state.pending_tasks.items():
            if future.done():
                try:
                    result = future.result()
                    completed_tasks[task_id] = {"agent": agent_id, "result": result}
                except Exception as e:
                    completed_tasks[task_id] = {"agent": agent_id, "error": str(e)}
            else:
                remaining_tasks[task_id] = (agent_id, future)
        
        # Update state
        state.completed_tasks.update(completed_tasks)
        state.pending_tasks = remaining_tasks
        
        # Check if all tasks completed
        if not remaining_tasks:
            return AggregateResults()
        
        return self  # Continue waiting
```

## Hierarchical State Machines

Complex workflows can be organized hierarchically with sub-SOPs.

### 1. Sub-SOP Pattern

```python
class MasterWorkflow(Procedure[MasterState]):
    @handler
    def handle_sub_workflow_needed(self, state: MasterState, event: SubWorkflowNeeded):
        workflow_type = event.payload.workflow_type
        
        # Create and execute sub-SOP
        if workflow_type == "data_processing":
            sub_sop = DataProcessingSOP(
                state=DataProcessingState(input_data=event.payload.data),
                init_proc=StartDataProcessing()
            )
        elif workflow_type == "report_generation":
            sub_sop = ReportGenerationSOP(
                state=ReportState(data=event.payload.data),
                init_proc=StartReportGeneration()
            )
        else:
            return f"Unknown sub-workflow type: {workflow_type}"
        
        # Execute sub-SOP
        with sub_sop:
            sub_result = execute_sub_workflow(sub_sop)
        
        # Integrate results back into master state
        state.sub_workflow_results[workflow_type] = sub_result
        
        return ContinueMasterWorkflow()

def execute_sub_workflow(sub_sop):
    """Execute a sub-SOP and return results"""
    # This would integrate with your agent execution logic
    # For now, simplified execution
    return {"status": "completed", "data": "sub_workflow_result"}
```

### 2. Nested State Management

```python
class HierarchicalState(BaseModel):
    # Master level state
    master_status: str = "initializing"
    current_sub_workflow: Optional[str] = None
    
    # Sub-workflow states
    data_processing_state: Optional[DataProcessingState] = None
    report_generation_state: Optional[ReportState] = None
    analysis_state: Optional[AnalysisState] = None
    
    # Results aggregation
    sub_workflow_results: dict = Field(default_factory=dict)
    
    def get_active_sub_state(self):
        """Get the currently active sub-state"""
        if self.current_sub_workflow == "data_processing":
            return self.data_processing_state
        elif self.current_sub_workflow == "report_generation":
            return self.report_generation_state
        elif self.current_sub_workflow == "analysis":
            return self.analysis_state
        return None
    
    def set_active_sub_state(self, workflow_type: str, state):
        """Set the active sub-state"""
        self.current_sub_workflow = workflow_type
        if workflow_type == "data_processing":
            self.data_processing_state = state
        elif workflow_type == "report_generation":
            self.report_generation_state = state
        elif workflow_type == "analysis":
            self.analysis_state = state
```

## Dynamic Workflow Generation

Generate workflows dynamically based on runtime conditions.

### 1. Workflow Builder Pattern

```python
class WorkflowBuilder:
    def __init__(self):
        self.procedures = []
        self.transitions = {}
    
    def add_procedure(self, procedure_class, condition=None):
        """Add a procedure to the workflow"""
        self.procedures.append((procedure_class, condition))
        return self
    
    def add_transition(self, from_proc, to_proc, event_type, condition=None):
        """Add a transition between procedures"""
        if from_proc not in self.transitions:
            self.transitions[from_proc] = []
        self.transitions[from_proc].append((to_proc, event_type, condition))
        return self
    
    def build(self, state, context):
        """Build the workflow based on current context"""
        # Filter procedures based on conditions
        active_procedures = []
        for proc_class, condition in self.procedures:
            if condition is None or condition(state, context):
                active_procedures.append(proc_class)
        
        # Create dynamic procedure classes with appropriate transitions
        return self._create_dynamic_workflow(active_procedures, state, context)
    
    def _create_dynamic_workflow(self, procedures, state, context):
        """Create dynamic procedure classes with runtime transitions"""
        # This would involve dynamic class creation and handler injection
        # Simplified implementation
        return procedures[0]() if procedures else None

# Usage
def create_adaptive_workflow(state, context):
    builder = WorkflowBuilder()
    
    # Add procedures based on context
    if context.requires_data_validation:
        builder.add_procedure(ValidateData)
    
    if context.requires_processing:
        builder.add_procedure(ProcessData)
    
    if context.requires_analysis:
        builder.add_procedure(AnalyzeData, 
                            condition=lambda s, c: c.analysis_enabled)
    
    if context.requires_reporting:
        builder.add_procedure(GenerateReport)
    
    return builder.build(state, context)
```

### 2. Template-Based Workflows

```python
class WorkflowTemplate:
    def __init__(self, template_config):
        self.config = template_config
    
    def instantiate(self, parameters):
        """Create a workflow instance from template"""
        procedures = []
        
        for step_config in self.config["steps"]:
            procedure_class = self._get_procedure_class(step_config["type"])
            
            # Customize procedure with parameters
            procedure = procedure_class()
            procedure.prompt = make_prompt(
                step_config["prompt_template"].format(**parameters)
            )
            
            # Add custom handlers based on configuration
            self._add_custom_handlers(procedure, step_config, parameters)
            
            procedures.append(procedure)
        
        return procedures[0] if procedures else None
    
    def _get_procedure_class(self, procedure_type):
        """Get procedure class by type"""
        procedure_map = {
            "data_collection": DataCollectionProcedure,
            "data_processing": DataProcessingProcedure,
            "analysis": AnalysisProcedure,
            "reporting": ReportingProcedure,
        }
        return procedure_map.get(procedure_type, GenericProcedure)
    
    def _add_custom_handlers(self, procedure, config, parameters):
        """Add custom handlers based on configuration"""
        # Dynamic handler creation based on configuration
        # This would involve runtime method creation
        pass

# Template configuration
workflow_template_config = {
    "name": "data_analysis_workflow",
    "steps": [
        {
            "type": "data_collection",
            "prompt_template": "Collect {data_type} data from {source}",
            "handlers": {
                "data_collected": "validate_and_proceed"
            }
        },
        {
            "type": "data_processing",
            "prompt_template": "Process the collected data using {algorithm}",
            "handlers": {
                "processing_complete": "analyze_results"
            }
        }
    ]
}
```

## Advanced Error Recovery

Sophisticated error recovery strategies for complex scenarios.

### 1. Multi-Level Recovery

```python
class MultiLevelRecovery(Procedure[RecoveryState]):
    def __init__(self):
        self.recovery_strategies = [
            self._quick_retry,
            self._fallback_method,
            self._alternative_approach,
            self._manual_intervention
        ]
        self.current_strategy = 0
    
    @handler
    def handle_operation_failed(self, state: RecoveryState, event: OperationFailed):
        failure = event.payload
        
        # Try recovery strategies in order
        if self.current_strategy < len(self.recovery_strategies):
            strategy = self.recovery_strategies[self.current_strategy]
            self.current_strategy += 1
            
            recovery_result = strategy(state, failure)
            if recovery_result.success:
                # Reset strategy counter on success
                self.current_strategy = 0
                return ContinueOperation()
            else:
                # Try next strategy
                return self.handle_operation_failed(state, event)
        else:
            # All strategies exhausted
            return "All recovery strategies failed. Manual intervention required."
    
    def _quick_retry(self, state, failure):
        """Simple retry strategy"""
        if failure.retryable and state.retry_count < 3:
            state.retry_count += 1
            return RecoveryResult(success=True, action="retry")
        return RecoveryResult(success=False)
    
    def _fallback_method(self, state, failure):
        """Try alternative method"""
        if state.has_fallback_available:
            state.using_fallback = True
            return RecoveryResult(success=True, action="fallback")
        return RecoveryResult(success=False)
    
    def _alternative_approach(self, state, failure):
        """Completely different approach"""
        if state.alternative_approaches:
            approach = state.alternative_approaches.pop(0)
            state.current_approach = approach
            return RecoveryResult(success=True, action="alternative")
        return RecoveryResult(success=False)
    
    def _manual_intervention(self, state, failure):
        """Request human intervention"""
        state.requires_manual_intervention = True
        return RecoveryResult(success=True, action="manual")
```

### 2. Predictive Recovery

```python
class PredictiveRecovery(Procedure[PredictiveState]):
    def __init__(self):
        self.health_metrics = {}
        self.failure_predictors = [
            self._check_resource_exhaustion,
            self._check_performance_degradation,
            self._check_error_rate_increase,
            self._check_external_dependencies
        ]
    
    @handler
    def handle_health_check(self, state: PredictiveState, event: HealthCheck):
        metrics = event.payload
        self.health_metrics.update(metrics)
        
        # Run failure predictors
        for predictor in self.failure_predictors:
            risk_level = predictor(self.health_metrics, state)
            
            if risk_level == "high":
                return PreventiveAction(predictor_name=predictor.__name__)
            elif risk_level == "medium":
                state.warning_count += 1
                if state.warning_count > 3:
                    return PreventiveAction(predictor_name=predictor.__name__)
        
        return self  # Continue monitoring
    
    def _check_resource_exhaustion(self, metrics, state):
        """Check if resources are running low"""
        if metrics.get("memory_usage", 0) > 0.9:
            return "high"
        elif metrics.get("memory_usage", 0) > 0.8:
            return "medium"
        return "low"
    
    def _check_performance_degradation(self, metrics, state):
        """Check if performance is degrading"""
        recent_response_times = metrics.get("response_times", [])[-10:]
        if recent_response_times:
            avg_time = sum(recent_response_times) / len(recent_response_times)
            if avg_time > state.performance_threshold * 2:
                return "high"
            elif avg_time > state.performance_threshold * 1.5:
                return "medium"
        return "low"
```

## Advanced State Patterns

Sophisticated state management for complex scenarios.

### 1. Versioned State

```python
class VersionedState(BaseModel):
    version: int = 1
    data: dict = Field(default_factory=dict)
    history: List[dict] = Field(default_factory=list)
    
    def update_data(self, new_data: dict, description: str = ""):
        """Update data with versioning"""
        # Save current state to history
        self.history.append({
            "version": self.version,
            "data": deepcopy(self.data),
            "timestamp": datetime.now(),
            "description": description
        })
        
        # Update to new version
        self.version += 1
        self.data.update(new_data)
    
    def rollback_to_version(self, target_version: int) -> bool:
        """Rollback to a specific version"""
        for historical_state in reversed(self.history):
            if historical_state["version"] == target_version:
                self.data = deepcopy(historical_state["data"])
                self.version = target_version
                return True
        return False
    
    def get_version_history(self) -> List[dict]:
        """Get version history"""
        return [
            {
                "version": h["version"],
                "timestamp": h["timestamp"],
                "description": h["description"]
            }
            for h in self.history
        ]
```

### 2. Distributed State

```python
class DistributedState(BaseModel):
    node_id: str
    local_data: dict = Field(default_factory=dict)
    shared_data: dict = Field(default_factory=dict)
    sync_status: dict = Field(default_factory=dict)
    
    def update_local(self, key: str, value: Any):
        """Update local data"""
        self.local_data[key] = value
        self.sync_status[key] = "pending_sync"
    
    def update_shared(self, key: str, value: Any, source_node: str):
        """Update shared data from another node"""
        self.shared_data[key] = {
            "value": value,
            "source_node": source_node,
            "timestamp": datetime.now()
        }
        self.sync_status[key] = "synced"
    
    def get_effective_value(self, key: str) -> Any:
        """Get the effective value (local takes precedence)"""
        if key in self.local_data:
            return self.local_data[key]
        elif key in self.shared_data:
            return self.shared_data[key]["value"]
        return None
    
    def needs_sync(self) -> List[str]:
        """Get keys that need synchronization"""
        return [
            key for key, status in self.sync_status.items()
            if status == "pending_sync"
        ]
```

## Performance Optimization

Patterns for optimizing SOPy performance in large-scale scenarios.

### 1. Lazy State Loading

```python
class LazyState(BaseModel):
    _loaded_sections: Set[str] = Field(default_factory=set, exclude=True)
    _data_loaders: Dict[str, Callable] = Field(default_factory=dict, exclude=True)
    
    def register_loader(self, section: str, loader: Callable):
        """Register a loader for a state section"""
        self._data_loaders[section] = loader
    
    def load_section(self, section: str):
        """Load a specific section of state"""
        if section not in self._loaded_sections and section in self._data_loaders:
            data = self._data_loaders[section]()
            setattr(self, section, data)
            self._loaded_sections.add(section)
    
    def __getattribute__(self, name):
        # Intercept attribute access to trigger lazy loading
        if name.startswith('_') or name in ['register_loader', 'load_section']:
            return super().__getattribute__(name)
        
        # Check if this section needs loading
        if (name not in self._loaded_sections and 
            name in self._data_loaders):
            self.load_section(name)
        
        return super().__getattribute__(name)
```

### 2. State Compression

```python
import pickle
import gzip

class CompressedState(BaseModel):
    _compressed_data: bytes = b""
    _is_compressed: bool = False
    
    def compress(self):
        """Compress the state data"""
        if not self._is_compressed:
            data = self.dict(exclude={"_compressed_data", "_is_compressed"})
            pickled_data = pickle.dumps(data)
            self._compressed_data = gzip.compress(pickled_data)
            self._is_compressed = True
            
            # Clear uncompressed data
            for field in data.keys():
                if hasattr(self, field):
                    delattr(self, field)
    
    def decompress(self):
        """Decompress the state data"""
        if self._is_compressed:
            decompressed_data = gzip.decompress(self._compressed_data)
            data = pickle.loads(decompressed_data)
            
            # Restore fields
            for key, value in data.items():
                setattr(self, key, value)
            
            self._is_compressed = False
            self._compressed_data = b""
```

## Summary

Advanced SOPy patterns enable:

- **Multi-agent coordination** for complex distributed tasks
- **Hierarchical workflows** with sub-SOPs and nested state management
- **Dynamic workflow generation** based on runtime conditions
- **Sophisticated error recovery** with multiple strategies and prediction
- **Advanced state management** with versioning, distribution, and optimization
- **Performance optimization** through lazy loading and compression

These patterns allow SOPy to handle enterprise-scale workflows with complex requirements while maintaining the framework's core principles of reliability, observability, and maintainability.

## Next Steps

- Apply these patterns to your specific use cases
- Combine multiple patterns for complex scenarios
- Contribute new patterns to the SOPy community
- Explore integration with other frameworks and systems

The advanced patterns in this tutorial represent the cutting edge of what's possible with SOPy. As you build more complex systems, you'll likely discover new patterns and techniques that can benefit the entire SOPy community.
