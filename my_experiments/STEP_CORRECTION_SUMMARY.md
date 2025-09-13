# Multi-Stage TIM Experiment: Step vs Epoch Correction

## Important Correction Made

**Original Issue**: The initial implementation incorrectly used **epochs** for TIM time windows instead of **training steps**.

**Correction**: TIM (Time-varying Influence Measurement) operates on **training steps**, where each step represents one batch gradient update.

## Key Changes Made

### 1. Time Window Calculation (`multi_stage_pruning_experiment.py`)

**Before** (Incorrect):
```python
# Divided epochs across stages
epochs_per_stage = total_epochs // num_stages
time_windows = [(start_epoch, end_epoch), ...]  # WRONG!
```

**After** (Correct):
```python
# Calculate total training steps
steps_per_epoch = (train_count + batch_size - 1) // batch_size
total_steps = total_epochs * steps_per_epoch

# Divide steps across stages
steps_per_stage = total_steps // num_stages
time_windows = [(start_step, end_step), ...]  # CORRECT!
```

### 2. Example Calculation

**Configuration**:
- Training samples: 1000
- Batch size: 16
- Total epochs: 10
- Number of stages: 5

**Step Calculation**:
```
Steps per epoch = ceil(1000/16) = 63 steps
Total steps = 10 × 63 = 630 steps
Steps per stage = 630 ÷ 5 = 126 steps

Time Windows (Steps):
Stage 1: [0, 125]     - 126 steps (~2.0 epochs)
Stage 2: [126, 251]   - 126 steps (~2.0 epochs)
Stage 3: [252, 377]   - 126 steps (~2.0 epochs)
Stage 4: [378, 503]   - 126 steps (~2.0 epochs)
Stage 5: [504, T]     - 126 steps (~2.0 epochs)
```

### 3. TIM Integration

**TIM Parameters** (Corrected):
```python
tim_evaluator = TimInfluence(
    start_step=t1,  # Training step (not epoch)
    end_step=t2,    # Training step (not epoch)
    # ... other parameters
)
```

### 4. Updated Documentation

- **README**: Clarified that time windows use steps, not epochs
- **Code comments**: Updated to specify "steps" vs "epochs"
- **Visualization**: Time window diagrams now show training steps
- **Output messages**: All references updated to "steps"

## Why This Matters

### 1. **Granularity**
- **Steps**: Fine-grained analysis (every batch update)
- **Epochs**: Coarse-grained analysis (full dataset passes)

### 2. **TIM Accuracy**
- TIM tracks influence at each gradient update (step)
- Using epochs would miss important temporal dynamics within epochs

### 3. **Temporal Sensitivity**
- Early steps within an epoch might behave differently than later steps
- Step-based windows capture this intra-epoch variation

## Verification

To verify correct implementation:

```python
# Run a small experiment and check output
experiment = create_multi_stage_experiment(
    train_count=100, batch_size=10, epochs=2, num_stages=2
)

# Expected output:
# Steps per epoch = ceil(100/10) = 10
# Total steps = 2 × 10 = 20
# Time windows: [0, 9], [10, T]
```

## Impact on Results

The correction ensures:

1. **Proper TIM computation**: Influence calculated for correct time intervals
2. **Meaningful comparisons**: Each stage analyzes equivalent numbers of gradient updates
3. **Temporal accuracy**: True step-by-step influence tracking as intended by TIM methodology

## Files Modified

1. `multi_stage_pruning_experiment.py` - Main experiment logic
2. `tim_influence_module.py` - Parameter documentation
3. `multi_stage_visualization_module.py` - Time window diagrams
4. `run_multi_stage_experiment.py` - Output display
5. `MULTI_STAGE_README.md` - Documentation
6. `STEP_CORRECTION_SUMMARY.md` - This summary

## Usage Note

When using the corrected experiment:

```bash
python run_multi_stage_experiment.py
```

You'll see output like:
```
📊 Time window calculation:
   Training samples: 1000
   Batch size: 16
   Steps per epoch: 63
   Total epochs: 10
   Total steps: 630
   Time windows (steps):
     Stage 1: [0, 125] - 126 steps (~2.0 epochs)
     Stage 2: [126, 251] - 126 steps (~2.0 epochs)
     # ... etc
```

This confirms the experiment is correctly using training steps for TIM time windows.