# Wah Loon Multi-Agent CAD System - Developer Guide

## Table of Contents
1. [Development Setup](#development-setup)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [Adding New Agents](#adding-new-agents)
4. [Customizing Workflows](#customizing-workflows)
5. [Testing Guidelines](#testing-guidelines)
6. [Deployment](#deployment)
7. [Troubleshooting](#troubleshooting)

---

## Development Setup

### Environment Setup

1. **Clone the Repository**
```bash
cd F:\AI_Coding\wahloon-multi-agent-cad
```

2. **Create Virtual Environment**
```bash
# Windows
python -m venv .myenv
.myenv\Scripts\activate

# Linux/Mac
python3 -m venv .myenv
source .myenv/bin/activate
```

3. **Install Dependencies**
```bash
# Install all dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-asyncio pytest-cov black flake8
```

4. **Configure Environment**

Create `.env` file:
```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.1
LOG_LEVEL=DEBUG
MAX_CONCURRENT_AGENTS=5
TIMEOUT_SECONDS=300
```

---

## Architecture Deep Dive

### LangGraph Integration

The system uses LangGraph's `StateGraph` to orchestrate agent execution:

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(MEAgentState)

# Add agent nodes
workflow.add_node("design_analyzer", self._run_design_analysis)
workflow.add_node("clash_detector", self._run_clash_detection)

# Define execution flow
workflow.set_entry_point("design_analyzer")
workflow.add_edge("design_analyzer", "clash_detector")

graph = workflow.compile()
```

### State Management Pattern

State flows through the workflow using TypedDict:

```python
class MEAgentState(TypedDict):
    # Input
    cad_data: Dict[str, Any]
    design_intent: Optional[str]
    
    # Outputs from each agent
    design_analysis: Optional[Dict]
    clash_analysis: Optional[Dict]
    
    # Metadata
    current_agent: str
    agent_progress: List[str]
    errors: List[str]
```

---

## Adding New Agents

### Step 1: Create Agent Class

Create new file `src/agents/cost_estimator.py`:

```python
from .agent_base import BaseMEAgent
from typing import Dict, Any
import json

class CostEstimatorAgent(BaseMEAgent):
    """Cost Estimation Specialist Agent"""
    
    def __init__(self, llm):
        super().__init__(
            llm=llm,
            agent_name="Cost Estimation Specialist",
            agent_role="Estimating project costs and budget analysis"
        )
    
    async def analyze(self, project_data: Dict, context: Dict = None) -> Dict[str, Any]:
        """Estimate project costs"""
        
        prompt = f"""
        Perform cost estimation for this M&E project:
        {json.dumps(project_data, indent=2)}
        
        Provide detailed cost estimation in JSON format...
        """
        
        response = await self._call_llm(prompt, context)
        analysis = self._extract_json_from_response(response)
        
        return {
            "agent": self.agent_name,
            "timestamp": self._create_timestamp(),
            "analysis": analysis,
            "summary": self._generate_summary(analysis)
        }
```

### Step 2: Update Agent Registry

In `src/agents/__init__.py`:

```python
from .cost_estimator import CostEstimatorAgent

__all__ = [
    'BaseMEAgent',
    'DesignAnalyzerAgent',
    'CostEstimatorAgent',  # Add new agent
    # ... others
]
```

### Step 3: Update State Schema

In `src/workflows/state_schema.py`:

```python
class MEAgentState(TypedDict):
    # ... existing fields
    cost_estimation: Optional[Dict[str, Any]]  # Add new field
```

### Step 4: Integrate into Workflow

In `src/workflows/multi_agent_workflow.py`:

```python
def _initialize_agents(self) -> Dict[str, Any]:
    return {
        # ... existing agents
        "cost_estimator": CostEstimatorAgent(self.llm),
    }

def _build_workflow(self) -> StateGraph:
    workflow = StateGraph(MEAgentState)
    
    # Add node
    workflow.add_node("cost_estimator", self._run_cost_estimation)
    
    # Add edge in appropriate position
    workflow.add_edge("energy_analyzer", "cost_estimator")
    workflow.add_edge("cost_estimator", "qa_reviewer")
    
    return workflow.compile()

# Add execution method
async def _run_cost_estimation(self, state: MEAgentState) -> Dict[str, Any]:
    """Execute cost estimation agent"""
    try:
        print("💰 [Cost Agent] Estimating project costs...")
        start_time = datetime.now()
        
        context = {
            "design_analysis": state.get("design_analysis"),
            "clash_analysis": state.get("clash_analysis")
        }
        
        result = await self.agents["cost_estimator"].analyze(
            state["cad_data"], context
        )
        
        updated_state = self.state_manager.update_agent_progress(
            state, "cost_estimator", {"cost_estimation": result}
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        updated_state["processing_times"]["cost_estimator"] = processing_time
        
        print(f"   ✅ Cost estimation completed in {processing_time:.2f}s")
        return updated_state
        
    except Exception as e:
        error_state = self.state_manager.add_error(state, str(e), "cost_estimator")
        print(f"   ❌ Cost estimation failed: {e}")
        return error_state
```

---

## Customizing Workflows

### Creating Conditional Workflows

Add conditional branching based on analysis results:

```python
def _build_workflow(self) -> StateGraph:
    workflow = StateGraph(MEAgentState)
    
    # Add nodes
    workflow.add_node("design_analyzer", self._run_design_analysis)
    workflow.add_node("quick_review", self._run_quick_review)
    workflow.add_node("detailed_review", self._run_detailed_review)
    
    # Set entry point
    workflow.set_entry_point("design_analyzer")
    
    # Add conditional edge
    workflow.add_conditional_edges(
        "design_analyzer",
        self._route_based_on_complexity,
        {
            "simple": "quick_review",
            "complex": "detailed_review"
        }
    )
    
    return workflow.compile()

def _route_based_on_complexity(self, state: MEAgentState) -> str:
    """Route based on project complexity"""
    design_analysis = state.get("design_analysis", {})
    
    if design_analysis.get("analysis", {}).get("completeness_assessment", {}).get("score", 0) > 80:
        return "simple"
    else:
        return "complex"
```

### Parallel Agent Execution

Execute independent agents in parallel:

```python
async def _run_parallel_analysis(self, state: MEAgentState) -> Dict[str, Any]:
    """Run clash and energy analysis in parallel"""
    
    design_context = {"design_analysis": state.get("design_analysis")}
    
    # Create tasks
    clash_task = self.agents["clash_detector"].analyze(
        state["cad_data"], design_context
    )
    energy_task = self.agents["energy_analyzer"].analyze(
        state["cad_data"], design_context
    )
    
    # Execute in parallel
    clash_result, energy_result = await asyncio.gather(
        clash_task, energy_task
    )
    
    # Update state with both results
    updated_state = state.copy()
    updated_state["clash_analysis"] = clash_result
    updated_state["energy_analysis"] = energy_result
    
    return updated_state
```

### Custom Agent Ordering

Create workflow variants for different project types:

```python
class CustomWorkflow(EnhancedMEAgentWorkflow):
    """Custom workflow for hospital projects"""
    
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(MEAgentState)
        
        # Hospital-specific flow
        workflow.add_node("design_analyzer", self._run_design_analysis)
        workflow.add_node("infection_control", self._run_infection_control_check)
        workflow.add_node("medical_systems", self._run_medical_systems_check)
        workflow.add_node("compliance_checker", self._run_compliance_check)
        workflow.add_node("qa_reviewer", self._run_qa_review)
        
        # Custom flow
        workflow.set_entry_point("design_analyzer")
        workflow.add_edge("design_analyzer", "infection_control")
        workflow.add_edge("infection_control", "medical_systems")
        workflow.add_edge("medical_systems", "compliance_checker")
        workflow.add_edge("compliance_checker", "qa_reviewer")
        workflow.add_edge("qa_reviewer", END)
        
        return workflow.compile()
```

---

## Testing Guidelines

### Unit Testing Agents

Create `tests/test_cost_estimator.py`:

```python
import pytest
from src.agents.cost_estimator import CostEstimatorAgent
from langchain_openai import ChatOpenAI

@pytest.fixture
def llm():
    """LLM fixture for testing"""
    return ChatOpenAI(model="gpt-4", temperature=0.1)

@pytest.fixture
def cost_agent(llm):
    """Cost estimator agent fixture"""
    return CostEstimatorAgent(llm)

@pytest.fixture
def sample_project_data():
    """Sample project data"""
    return {
        "project_id": "TEST-001",
        "project_name": "Test Project",
        "total_area_sqm": 10000,
        "electrical_system": {
            "main_switchboard": "1600A, 22kV"
        },
        "mechanical_system": {
            "chiller_capacity_tr": 800
        }
    }

@pytest.mark.asyncio
async def test_cost_estimation(cost_agent, sample_project_data):
    """Test basic cost estimation"""
    result = await cost_agent.analyze(sample_project_data)
    
    assert result is not None
    assert 'agent' in result
    assert 'analysis' in result
    assert 'summary' in result
    assert result['agent'] == "Cost Estimation Specialist"

@pytest.mark.asyncio
async def test_cost_estimation_with_context(cost_agent, sample_project_data):
    """Test cost estimation with context"""
    context = {
        "design_analysis": {
            "analysis": {
                "technical_risks": [
                    {"severity": "HIGH", "impact": "cost"}
                ]
            }
        }
    }
    
    result = await cost_agent.analyze(sample_project_data, context)
    
    assert result is not None
    assert 'analysis' in result

@pytest.mark.asyncio
async def test_cost_summary_generation(cost_agent, sample_project_data):
    """Test summary generation"""
    result = await cost_agent.analyze(sample_project_data)
    summary = result.get('summary', '')
    
    assert summary is not None
    assert len(summary) > 0
```

### Integration Testing

Create `tests/test_workflow_integration.py`:

```python
import pytest
import asyncio
from src.workflows.multi_agent_workflow import EnhancedMEAgentWorkflow
from src.data.generators import WahLoonDataGenerator
from langchain_openai import ChatOpenAI

@pytest.fixture
def workflow():
    """Workflow fixture"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.1)
    return EnhancedMEAgentWorkflow(llm)

@pytest.fixture
def project_data():
    """Generate test project data"""
    generator = WahLoonDataGenerator()
    return generator.generate_cad_project("TEST-WORKFLOW-001")

@pytest.mark.asyncio
async def test_complete_workflow(workflow, project_data):
    """Test complete workflow execution"""
    results = await workflow.run_workflow(project_data)
    
    # Verify all agents completed
    assert len(results['agent_progress']) >= 5
    
    # Verify all analysis results present
    assert 'design_analysis' in results
    assert 'clash_analysis' in results
    assert 'compliance_check' in results
    assert 'energy_analysis' in results
    assert 'qa_report' in results
    
    # Verify final report
    assert 'final_report' in results
    assert results['final_report'] is not None

@pytest.mark.asyncio
async def test_workflow_error_handling(workflow, project_data):
    """Test workflow error handling"""
    # Modify data to potentially cause errors
    project_data['electrical_system'] = None
    
    results = await workflow.run_workflow(project_data)
    
    # Workflow should complete even with errors
    assert results is not None
    # Check if errors were recorded
    assert isinstance(results.get('errors', []), list)

@pytest.mark.asyncio
async def test_workflow_with_design_intent(workflow, project_data):
    """Test workflow with design intent"""
    design_intent = "Sustainable design for BCA Green Mark Platinum"
    
    results = await workflow.run_workflow(project_data, design_intent)
    
    assert results is not None
    assert results.get('design_intent') == design_intent
```

### Performance Testing

Create `tests/test_performance.py`:

```python
import pytest
import time
from src.workflows.multi_agent_workflow import EnhancedMEAgentWorkflow
from src.data.generators import WahLoonDataGenerator

@pytest.mark.asyncio
async def test_workflow_performance(workflow):
    """Test workflow completes within acceptable time"""
    generator = WahLoonDataGenerator()
    project_data = generator.generate_cad_project()
    
    start_time = time.time()
    results = await workflow.run_workflow(project_data)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    # Assert workflow completes within 2 minutes
    assert execution_time < 120, f"Workflow took {execution_time:.2f}s (max 120s)"
    
    # Verify processing times are recorded
    assert 'processing_times' in results
    assert 'total' in results['processing_times']

@pytest.mark.asyncio
async def test_batch_processing_performance(workflow):
    """Test batch processing performance"""
    generator = WahLoonDataGenerator()
    project_count = 3
    
    start_time = time.time()
    
    for i in range(project_count):
        project_data = generator.generate_cad_project(f"BATCH-{i:03d}")
        await workflow.run_workflow(project_data)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / project_count
    
    print(f"Batch processed {project_count} projects in {total_time:.2f}s")
    print(f"Average time per project: {avg_time:.2f}s")
    
    # Each project should complete within 90 seconds on average
    assert avg_time < 90
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_agents.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run tests matching pattern
pytest tests/ -k "cost_estimator"

# Run async tests only
pytest tests/ -m asyncio

# Verbose output
pytest tests/ -v

# Show print statements
pytest tests/ -s
```

---

## Deployment

### Production Configuration

Create `config/production.env`:

```env
OPENAI_API_KEY=prod_key_here
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.05
LOG_LEVEL=INFO
MAX_CONCURRENT_AGENTS=10
TIMEOUT_SECONDS=600
```

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY main.py .
COPY .env .

# Run application
CMD ["python", "main.py", "--mode", "single"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  wahloon-cad-analyzer:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_MODEL=gpt-4
      - LOG_LEVEL=INFO
    volumes:
      - ./output:/app/output
    restart: unless-stopped
```

Build and run:

```bash
docker-compose up --build
```

### API Service Deployment

Create `api_server.py`:

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import asyncio
from src.workflows.multi_agent_workflow import EnhancedMEAgentWorkflow
from langchain_openai import ChatOpenAI
from src.utils.config import Config

app = FastAPI(title="Wah Loon CAD Analysis API")

# Initialize workflow
llm = ChatOpenAI(**Config.get_llm_config())
workflow = EnhancedMEAgentWorkflow(llm)

class AnalysisRequest(BaseModel):
    project_data: dict
    design_intent: str = None

class AnalysisResponse(BaseModel):
    status: str
    task_id: str

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_project(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Submit project for analysis"""
    task_id = f"task_{asyncio.get_event_loop().time()}"
    
    background_tasks.add_task(
        run_analysis,
        task_id,
        request.project_data,
        request.design_intent
    )
    
    return AnalysisResponse(
        status="queued",
        task_id=task_id
    )

async def run_analysis(task_id: str, project_data: dict, design_intent: str):
    """Run analysis in background"""
    results = await workflow.run_workflow(project_data, design_intent)
    # Save results...
    
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Troubleshooting

### Common Issues

#### 1. OpenAI API Errors

**Problem:** `openai.error.RateLimitError`

**Solution:**
```python
# Add retry logic
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(3))
async def _call_llm_with_retry(self, prompt: str):
    return await self._call_llm(prompt)
```

#### 2. State Management Issues

**Problem:** State not updating between agents

**Solution:**
```python
# Ensure state is properly copied
def update_state(state: MEAgentState, updates: Dict) -> MEAgentState:
    new_state = state.copy()  # Important!
    new_state.update(updates)
    return new_state
```

#### 3. JSON Parsing Errors

**Problem:** LLM returns invalid JSON

**Solution:**
```python
def _extract_json_from_response(self, response: str) -> Dict:
    """Extract JSON with fallback"""
    try:
        # Try to find JSON block
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = response[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
        # Return structured fallback
        return {
            "raw_response": response,
            "parse_error": str(e)
        }
```

#### 4. Timeout Issues

**Problem:** Agents taking too long

**Solution:**
```python
import asyncio

async def _run_with_timeout(self, coro, timeout=300):
    """Run coroutine with timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"Operation timed out after {timeout}s")
        raise
```

### Debugging Tips

#### Enable Verbose Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

#### Inspect State at Each Step

```python
async def _run_design_analysis(self, state: MEAgentState):
    print(f"Input state keys: {state.keys()}")
    result = await self.agents["design_analyzer"].analyze(state["cad_data"])
    print(f"Result keys: {result.keys()}")
    return result
```

#### Use Breakpoints

```python
import pdb

async def _run_workflow(self, data):
    pdb.set_trace()  # Debugger will stop here
    results = await self.graph.ainvoke(data)
    return results
```

---

## Best Practices

### 1. Agent Design

- **Single Responsibility:** Each agent should focus on one specific task
- **Clear Outputs:** Define clear JSON schemas for agent outputs
- **Context Awareness:** Use context from previous agents effectively
- **Error Handling:** Always handle exceptions gracefully

### 2. Prompt Engineering

- **Be Specific:** Provide clear instructions and expected formats
- **Include Examples:** Show the LLM what good output looks like
- **Constrain Outputs:** Request JSON format with specific schema
- **Add Domain Context:** Include Singapore standards and practices

### 3. State Management

- **Immutable Updates:** Always create new state copies
- **Type Safety:** Use TypedDict for type checking
- **Clear Naming:** Use descriptive field names
- **Documentation:** Document state transitions

### 4. Performance Optimization

- **Reuse LLM Instances:** Don't recreate for each call
- **Batch When Possible:** Process multiple items together
- **Cache Results:** Cache expensive computations
- **Async Operations:** Use async/await consistently

---

## Contributing

### Code Style

Follow PEP 8 guidelines:

```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

### Commit Guidelines

```
feat: Add cost estimation agent
fix: Resolve JSON parsing issue in clash detector
docs: Update API reference
test: Add integration tests for workflow
refactor: Simplify state management logic
```

### Pull Request Process

1. Create feature branch
2. Write tests for new functionality
3. Update documentation
4. Run full test suite
5. Submit PR with description

---

**For additional support, refer to README.md and API_REFERENCE.md**
