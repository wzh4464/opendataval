# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenDataVal is a unified benchmark for data valuation algorithms. It's a Python library that provides standardized datasets, data valuation methods, and evaluation tasks for image, NLP, and tabular data.

## Development Commands

### Installation

```bash
# Production installation
make install

# Development installation (includes pre-commit hooks)
make install-dev
```

### Testing and Quality Checks

```bash
# Run tests with coverage
make coverage

# Format code with ruff
make format

# Clean temporary files
make clean
```

### Build Commands

```bash
# Update dependencies
make build
```

### CLI Usage

```bash
# Use the CLI tool
opendataval --file cli.csv -n [job_id] -o [path/to/output/]

# Or without installation
python opendataval --file cli.csv -n [job_id] -o [path/to/output/]
```

## Architecture and Components

The library consists of 4 main interacting parts:

### 1. DataFetcher (`opendataval.dataloader`)

- Loads and preprocesses datasets
- Handles data splits and noise injection
- Available datasets can be queried with `DataFetcher.datasets_available()`
- Supports embeddings for image/NLP datasets

### 2. Model (`opendataval.model`)

- Provides trainable prediction models
- Supports various architectures: LogisticRegression, MLP, LeNet, BERT
- Compatible with scikit-learn models
- Models follow PyTorch-style API (`fit`, `predict`)

### 3. DataEvaluator (`opendataval.dataval`)

- Implements different data valuation algorithms
- Available methods: AME, CS-Shapley, DVRL, Influence Functions, KNN-Shapley, LAVA, Leave-One-Out, Random, Volume-based methods
- All evaluators inherit from the `DataEvaluator` base class
- Common pattern: `evaluator.input_data().train_data_values().evaluate_data_values()`

### 4. ExperimentMediator (`opendataval.experiment`)

- Orchestrates experiments across multiple data evaluators
- Handles model training and evaluation
- Provides both plotting and non-plotting experiment methods
- Factory method: `ExperimentMediator.model_factory_setup()`

## Key Design Patterns

### Data Pipeline

```python
# Standard workflow
fetcher = DataFetcher(dataset_name='iris')
model = LogisticRegression(input_dim, output_dim)
evaluator = DataOob()

# Train and evaluate
dataval = evaluator.train(fetcher, model, metric='accuracy')
data_values = dataval.evaluate_data_values()
```

### Experiment Setup

```python
# Using ExperimentMediator
exper_med = ExperimentMediator.model_factory_setup(
    dataset_name='iris',
    train_count=50,
    valid_count=50,
    test_count=50,
    model_name='ClassifierMLP'
)
eval_med = exper_med.compute_data_values([DataOob()])
```

## Code Quality Standards

- Uses ruff for linting and formatting (line length: 88)
- Follows numpy docstring convention
- Test coverage target: ~75%
- Pre-commit hooks enforce code quality
- Supports Python 3.9-3.11

## Testing

- Tests are in the `test/` directory
- Uses pytest with coverage reporting
- CI/CD runs tests on Ubuntu and Windows
- Single test files can be run: `pytest test/test_specific.py`

## Dependencies

Key dependencies are version-pinned for stability:

- PyTorch (~2.2.2) for deep learning models
- scikit-learn (~1.3) for traditional ML
- numpy (~1.26.4) for numerical computing
- pandas for data manipulation
- transformers (~4.38) for NLP models

## Documentation

- Sphinx-based documentation at <https://opendataval.github.io>
- README files in each major package directory provide specific guidance
- Examples in the `examples/` directory demonstrate usage patterns
