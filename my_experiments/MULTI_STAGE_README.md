# Multi-Stage TIM Influence Data Pruning Experiment

This experiment implements a comprehensive multi-stage approach to TIM (Time-varying Influence Measurement) data valuation and noise pruning for BERT sentiment analysis.

## Overview

The experiment divides the training process into **5 time windows** and performs TIM influence computation and data cleaning for each stage:

- **Stage 1**: `[0, t1]` - Early training phase
- **Stage 2**: `[t1, t2]` - Initial convergence
- **Stage 3**: `[t2, t3]` - Mid training
- **Stage 4**: `[t3, t4]` - Late training  
- **Stage 5**: `[t4, T]` - Final convergence

For each stage, the experiment:
1. Computes TIM influence scores for that specific time window
2. Identifies and prunes low-influence samples (likely noise)
3. Retrains the model on cleaned data with the same initialization
4. Compares performance across all stages

## Key Features

### 🎯 **Multi-Stage Analysis**
- Temporal sensitivity analysis of data importance
- Stage-wise noise detection effectiveness
- Time window optimization for data cleaning

### 📊 **Comprehensive Visualization**
- **All text labels and outputs in English**
- Multi-stage influence score comparisons
- Performance metrics across time windows
- Training curves comparison (original vs cleaned models)
- Noise detection effectiveness analysis
- Interactive experiment dashboard
- Time window division diagrams

### 🔬 **Robust Experimental Design**
- Consistent model initialization across stages
- Controlled noise injection (30% label noise by default)
- Statistical analysis of results
- Error handling and experiment reproducibility

## Quick Start

### Run the Demonstration

```bash
cd my_experiments
python run_multi_stage_experiment.py
```

This will run a complete 5-stage experiment with optimized settings:
- Dataset: IMDB sentiment analysis
- Training samples: 1,000
- Noise rate: 30%
- Total epochs: 10 (2 epochs per stage)
- Output: `./multi_stage_demo_results/`

### Custom Configuration

```python
from my_experiments.multi_stage_pruning_experiment import create_multi_stage_experiment

# Create custom experiment
experiment = create_multi_stage_experiment(
    dataset_name="imdb",
    train_count=2000,        # More samples
    noise_rate=0.4,          # 40% noise
    epochs=15,               # 3 epochs per stage
    num_stages=5,            # 5 time windows
    output_dir="./my_results"
)

# Run experiment
results = experiment.run_complete_experiment()
```

## Experiment Architecture

### Core Components

1. **MultiStagePruningExperiment** (`multi_stage_pruning_experiment.py`)
   - Main experiment orchestrator
   - Handles multi-stage workflow
   - Manages time window division

2. **MultiStageVisualizer** (`multi_stage_visualization_module.py`)
   - Comprehensive visualization suite
   - All labels and text in English
   - Interactive dashboard creation

3. **Supporting Modules**
   - `noise_data_module.py` - Noise injection and data processing
   - `tim_influence_module.py` - TIM computation and analysis
   - `bert_training_module.py` - BERT model training

### Time Window Calculation

The experiment automatically divides the total training epochs into equal time windows:

```
Total epochs = 10, Stages = 5
Stage 1: [0, 2]   - Epochs 0-2
Stage 2: [2, 4]   - Epochs 2-4  
Stage 3: [4, 6]   - Epochs 4-6
Stage 4: [6, 8]   - Epochs 6-8
Stage 5: [8, T]   - Epochs 8-10 (to end)
```

## Generated Outputs

### Results Files
- `multi_stage_experiment_results.json` - Complete results with metrics
- `experiment_config.json` - Configuration used
- `stage_*/` - Individual stage results and models

### Visualization Files
- `multi_stage_influence_comparison.png` - Influence distributions across stages
- `stage_performance_comparison.png` - Performance metrics by time window
- `training_curves_comparison.png` - Loss/accuracy curves comparison
- `influence_heatmap.png` - Heatmap of influence patterns
- `time_window_diagram.png` - Visual representation of time division
- `comprehensive_dashboard.png` - Complete experiment overview

## Key Insights

### 🔍 **Temporal Sensitivity**
Different time windows excel at identifying different types of problematic samples:
- **Early stages** (`[0, t1]`): Catch obvious label errors
- **Mid stages** (`[t2, t3]`): Identify harder cases during learning
- **Late stages** (`[t4, T]`): Find subtle inconsistencies during convergence

### 📈 **Performance Analysis**
- Compare effectiveness across time windows
- Identify optimal pruning strategies
- Analyze noise detection patterns over training

### 💡 **Practical Applications**
- Time-aware data cleaning strategies
- Multi-stage ensemble approaches
- Temporal robustness assessment

## Advanced Usage

### Batch Processing

```python
# Run multiple configurations
configs = [
    {"noise_rate": 0.2, "epochs": 10},
    {"noise_rate": 0.3, "epochs": 15}, 
    {"noise_rate": 0.4, "epochs": 20}
]

results = []
for config in configs:
    experiment = create_multi_stage_experiment(**config)
    result = experiment.run_complete_experiment()
    results.append(result)
```

### Custom Time Windows

```python
class CustomMultiStageExperiment(MultiStagePruningExperiment):
    def _calculate_time_windows(self, total_epochs, num_stages):
        # Custom time window logic
        return [(0, 3), (3, 7), (7, 12), (12, 18), (18, None)]
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers library
- OpenDataVal framework
- Matplotlib, Seaborn (for visualization)

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dataset_name` | Dataset to use | `"imdb"` |
| `train_count` | Training samples | `1000` |
| `noise_rate` | Label noise ratio | `0.3` |
| `epochs` | Total training epochs | `10` |
| `num_stages` | Number of time windows | `5` |
| `prune_ratio` | Pruning ratio per stage | `0.3` |
| `model_name` | BERT model variant | `"distilbert-base-uncased"` |
| `batch_size` | Training batch size | `16` |
| `random_state` | Random seed | `42` |

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce `train_count` and `batch_size`
   - Use CPU-only mode: `device="cpu"`

2. **Long Runtime**
   - Reduce `epochs` and `num_stages`
   - Use smaller dataset

3. **Visualization Errors**
   - Install missing packages: `pip install seaborn matplotlib`
   - Set `save_plots=False` to skip plots

### Debug Mode

```python
experiment = create_multi_stage_experiment(
    train_count=100,     # Minimal dataset
    epochs=2,            # Quick training
    num_stages=2,        # Fewer stages
    save_plots=False     # Skip visualization
)
```

## Citation

If you use this multi-stage TIM experiment in your research, please cite:

```bibtex
@misc{multistage_tim_experiment,
  title={Multi-Stage TIM Influence Data Pruning for BERT},
  author={Your Name},
  year={2024},
  howpublished={https://github.com/your-repo/opendataval}
}
```

## Contact

For questions or issues, please open an issue in the repository or contact the development team.

---

**Note**: This experiment is designed for research purposes and demonstration of multi-stage data valuation techniques. For production use, consider computational requirements and scaling factors.