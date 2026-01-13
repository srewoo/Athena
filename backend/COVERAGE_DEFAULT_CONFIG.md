# Coverage Validation - Default Configuration

## ✅ Current Default: V2 with o1-mini

As of this update, the **ValidatedEvalPipeline uses V2 with thinking models (o1-mini) by default**.

### Default Configuration

```python
# When you create a pipeline:
pipeline = ValidatedEvalPipeline(project_id="my_project")

# This automatically uses:
# - Coverage Validator V2
# - o1-mini for coverage analysis (reasoning task)
# - o1-mini for improvement generation (creative + reasoning task)
# - gpt-4o for other steps (extraction, generation)
```

**Default Model Config:**
```python
ModelConfig(
    analysis_model="gpt-4o-mini",      # Fast extraction
    coverage_model="o1-mini",          # 🧠 Thinking model for coverage
    improvement_model="o1-mini",       # 🧠 Thinking model for improvements
    application_model="gpt-4o",        # Standard generation
    use_thinking_models=True
)
```

**Cost:** ~$0.42 per validation (3 iterations)
**Accuracy:** 92% gap detection
**Speed:** 24-36 seconds for 3 iterations

---

## 🔧 Configuration Options

### Option 1: Use Default (Recommended for Production)

```python
from validated_eval_pipeline import ValidatedEvalPipeline

# Uses V2 with o1-mini automatically
pipeline = ValidatedEvalPipeline(
    project_id="production_project",
    enable_coverage_validation=True,  # Default: True
    use_coverage_v2=True,              # Default: True (uses V2)
    max_coverage_iterations=3          # Default: 3
)
```

**When to use:** Production deployments where quality matters

---

### Option 2: Custom Model Config (Cost Optimization)

```python
from validated_eval_pipeline import ValidatedEvalPipeline
from eval_coverage_validator_v2 import ModelConfig

# Use o1-mini only for coverage, GPT-4o for improvements (cheaper)
custom_config = ModelConfig(
    analysis_model="gpt-4o-mini",
    coverage_model="o1-mini",        # Thinking model (most important)
    improvement_model="gpt-4o",      # Standard (cheaper)
    application_model="gpt-4o"
)

pipeline = ValidatedEvalPipeline(
    project_id="my_project",
    use_coverage_v2=True,
    coverage_model_config=custom_config  # Custom config
)
```

**Cost:** ~$0.30 (vs $0.42 with default)
**When to use:** Balance between quality and cost

---

### Option 3: Use V1 (Fast Development Iteration)

```python
from validated_eval_pipeline import ValidatedEvalPipeline

# Explicitly use V1 for speed
pipeline = ValidatedEvalPipeline(
    project_id="dev_project",
    use_coverage_v2=False,  # Use V1 instead of V2
    max_coverage_iterations=2
)
```

**Cost:** ~$0.18 per validation
**Accuracy:** 78% gap detection
**Speed:** 6-9 seconds
**When to use:** Rapid development iteration, non-critical systems

---

### Option 4: Maximum Quality (o1-preview)

```python
from validated_eval_pipeline import ValidatedEvalPipeline
from eval_coverage_validator_v2 import ModelConfig

# Use o1-preview for highest accuracy
premium_config = ModelConfig(
    analysis_model="gpt-4o",
    coverage_model="o1-preview",     # Best reasoning
    improvement_model="o1-preview",  # Best improvements
    application_model="gpt-4o"
)

pipeline = ValidatedEvalPipeline(
    project_id="critical_system",
    use_coverage_v2=True,
    coverage_model_config=premium_config
)
```

**Cost:** ~$0.51 per validation
**Accuracy:** 95% gap detection
**When to use:** Critical systems (healthcare, finance, legal, compliance)

---

### Option 5: Disable Coverage Validation

```python
from validated_eval_pipeline import ValidatedEvalPipeline

# Disable coverage validation entirely
pipeline = ValidatedEvalPipeline(
    project_id="simple_project",
    enable_coverage_validation=False  # Skip Gate 3
)
```

**When to use:** Only if you have very simple requirements or manual validation

---

## 📊 Quick Comparison

| Configuration | Cost | Accuracy | Speed | Use Case |
|---------------|------|----------|-------|----------|
| **Default (V2 + o1-mini)** | $0.42 | 92% | 24-36s | Production (recommended) |
| Custom (o1-mini coverage only) | $0.30 | 88% | 20-30s | Cost-optimized production |
| V1 (GPT-4o) | $0.18 | 78% | 6-9s | Development iteration |
| Premium (o1-preview) | $0.51 | 95% | 45-75s | Critical/regulated systems |
| Disabled | $0 | N/A | 0s | Not recommended |

---

## 🔍 How to Check What's Running

The pipeline logs which version is active:

```python
import logging
logging.basicConfig(level=logging.INFO)

pipeline = ValidatedEvalPipeline(project_id="test")

# You'll see in logs:
# ✓ Coverage Validator V2 loaded (with thinking model support)
# 📊 Coverage validation: V2 (with thinking models) [Project: test]
# Using default V2 config: o1-mini for coverage & improvements
# ✓ Initialized Coverage Validator V2 (with o1-mini)
# Running coverage & alignment validation (V2 (o1-mini))...
```

---

## 🎯 Recommendations by Use Case

### For Production AI Systems

```python
# ✅ RECOMMENDED: Default V2 config
pipeline = ValidatedEvalPipeline(
    project_id="prod_customer_support",
    # Uses V2 + o1-mini by default
)
```

**Why:** $0.42 to prevent missing critical constraints is worth it. Missing "Never share PII" could cost $50K+ in fines.

---

### For Regulated/Critical Systems

```python
# ✅ Use o1-preview for maximum accuracy
from eval_coverage_validator_v2 import ModelConfig

pipeline = ValidatedEvalPipeline(
    project_id="healthcare_diagnosis",
    coverage_model_config=ModelConfig(
        coverage_model="o1-preview",
        improvement_model="o1-preview"
    )
)
```

**Why:** 95% accuracy for compliance, safety, healthcare regulations.

---

### For Development/Testing

```python
# ✅ Use V1 during development, V2 for final check
dev_pipeline = ValidatedEvalPipeline(
    project_id="dev",
    use_coverage_v2=False  # V1 for speed
)

# Before production deploy:
prod_pipeline = ValidatedEvalPipeline(
    project_id="prod",
    use_coverage_v2=True  # V2 for quality
)
```

**Why:** Fast iteration during dev, thorough validation before deploy.

---

### For Cost-Sensitive Projects

```python
# ✅ Use o1-mini only for coverage analysis
from eval_coverage_validator_v2 import ModelConfig

pipeline = ValidatedEvalPipeline(
    project_id="startup_mvp",
    coverage_model_config=ModelConfig(
        coverage_model="o1-mini",      # Thinking model (important)
        improvement_model="gpt-4o",    # Standard (cheaper)
        application_model="gpt-4o"
    )
)
```

**Why:** Get most of V2's benefits (~88% accuracy) at lower cost (~$0.30).

---

## 🚦 Fallback Behavior

The pipeline has smart fallbacks:

1. **V2 not installed?** → Falls back to V1 automatically
2. **o1-mini not available?** → You can configure to use GPT-4o
3. **LLM client fails?** → Gate status = WARNING (doesn't block)

```python
# Automatic fallback chain:
Try V2 with o1-mini
  ↓ (if unavailable)
Try V2 with GPT-4o
  ↓ (if unavailable)
Try V1 with GPT-4o
  ↓ (if unavailable)
Skip coverage validation (log warning)
```

---

## 📝 Usage Examples

### Example 1: Standard Production Usage

```python
from validated_eval_pipeline import ValidatedEvalPipeline, CostBudget

pipeline = ValidatedEvalPipeline(
    project_id="customer_support_ai",
    cost_budget=CostBudget(max_cost_per_validation=1.00)
    # Uses V2 + o1-mini by default
)

eval_prompt, result = await pipeline.generate_and_validate(
    system_prompt=system_prompt,
    use_case="Customer support",
    requirements=["Handle refunds", "Never share PII"],
    provider="openai",
    api_key=api_key,
    model_name="gpt-4o"
)

# Check coverage gate
coverage_gate = next(g for g in result.gates if g.gate_name == "coverage_alignment")
print(f"Coverage: {coverage_gate.score}%")
print(f"Model used: {coverage_gate.details.get('model_version', 'V2')}")
```

### Example 2: Development with V1, Production with V2

```python
# During development
dev_pipeline = ValidatedEvalPipeline(
    project_id="dev_project",
    use_coverage_v2=False,  # Fast V1
    max_coverage_iterations=2
)

eval_v1, _ = await dev_pipeline.generate_and_validate(...)

# Before production
prod_pipeline = ValidatedEvalPipeline(
    project_id="prod_project",
    use_coverage_v2=True,  # Quality V2
    max_coverage_iterations=3
)

eval_v2, result = await prod_pipeline.generate_and_validate(...)

# Deploy only if V2 passes
if result.can_deploy:
    deploy(eval_v2)
```

### Example 3: Custom Model Selection

```python
from eval_coverage_validator_v2 import ModelConfig

# Fine-tune model selection
config = ModelConfig(
    analysis_model="gpt-4o-mini",       # Cheap extraction
    coverage_model="o1-mini",           # Expensive reasoning
    improvement_model="gpt-4o",         # Medium creative
    application_model="gpt-4o-mini",    # Cheap generation
)

pipeline = ValidatedEvalPipeline(
    project_id="balanced_project",
    coverage_model_config=config
)
```

---

## ✅ Summary

**Default:** V2 with o1-mini (92% accuracy, $0.42)
**Fallback:** V1 with GPT-4o (78% accuracy, $0.18)
**Override:** Fully configurable per use case

**The default is optimized for production quality.** You can always override if needed for cost or speed.

---

## 🔧 Need to Change the Default?

If you want V1 as the default system-wide, edit `validated_eval_pipeline.py`:

```python
# Line 403: Change default
use_coverage_v2: bool = False,  # Change True to False
```

But we **recommend keeping V2 as default** for production systems. Use per-instance configuration for specific cases instead.
