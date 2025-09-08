#!/usr/bin/env python3
"""
🎯 BERT Sentiment Analysis with Influence Function Detection and Retraining Pipeline
================================================================================
Complete pipeline: 
1. Train original BERT model on noisy data
2. Use Influence Function to detect high-value samples
3. Select top 90% high-influence samples
4. Retrain BERT model on cleaned data
5. Compare original vs cleaned model performance
"""

import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score

from opendataval.dataloader import DataFetcher
from opendataval.model import BertClassifier, LogisticRegression
from opendataval.dataval import InfluenceFunction

def calculate_accuracy(model, x_data, y_true):
    """Calculate model accuracy"""
    with torch.no_grad():
        predictions = model.predict(x_data)
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        
        # If predictions are probabilities, convert to class predictions
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)
        
        # If y_true is one-hot encoded, convert to class labels
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        
        # Ensure both are 1D arrays
        predictions = predictions.flatten()
        y_true = y_true.flatten()
        
        return accuracy_score(y_true, predictions)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="BERT Influence Function Detection and Retraining Pipeline")
    parser.add_argument("--train_count", type=int, default=2048, help="Number of training samples")
    parser.add_argument("--test_count", type=int, default=256, help="Number of test samples")
    parser.add_argument("--noise_rate", type=float, default=0.3, help="Label noise rate (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--select_ratio", type=float, default=0.9, help="Ratio of high-influence samples to select")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs for BERT")
    parser.add_argument("--output_dir", type=str, default="./bert_influence_retrain_results", help="Output directory")
    return parser.parse_args()

def setup_data_and_noise(train_count, test_count, noise_rate, seed):
    """Setup data with noise injection"""
    print(f"🔄 Setting up data...")
    print(f"   Dataset: IMDB")
    print(f"   Training samples: {train_count}")
    print(f"   Test samples: {test_count}")
    print(f"   Noise rate: {noise_rate*100:.1f}%")
    
    # Create data fetcher with noise
    data_fetcher = (
        DataFetcher("imdb", cache_dir="../data_files/", force_download=False, random_state=seed)
        .split_dataset_by_count(train_count, test_count, test_count)
    )
    
    if noise_rate > 0:
        data_fetcher = data_fetcher.noisify("mix_labels", noise_rate=noise_rate)
        print(f"✅ Label flipping complete (30% labels flipped)")
    
    return data_fetcher

def train_original_bert_model(data_fetcher, epochs=5):
    """Train original BERT model on noisy data"""
    print(f"\n🤖 Training Original BERT Model (with noise)")
    print(f"   Model: DistilBERT-base-uncased")
    print(f"   Epochs: {epochs}")
    
    # Create BERT model
    bert_model = BertClassifier(
        pretrained_model_name="distilbert-base-uncased",
        num_classes=2
    )
    
    start_time = time.time()
    
    # Train the model
    bert_model.fit(
        data_fetcher.x_train, 
        data_fetcher.y_train,
        epochs=epochs,
        batch_size=16
    )
    
    training_time = time.time() - start_time
    
    # Evaluate original model
    train_acc = calculate_accuracy(bert_model, data_fetcher.x_train, data_fetcher.y_train)
    test_acc = calculate_accuracy(bert_model, data_fetcher.x_test, data_fetcher.y_test)
    
    print(f"✅ Original BERT training complete")
    print(f"   Training time: {training_time:.1f}s")
    print(f"   Training accuracy: {train_acc:.4f}")
    print(f"   Test accuracy: {test_acc:.4f}")
    
    return bert_model, {
        "training_time": training_time,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc
    }

def compute_influence_scores(data_fetcher):
    """Compute influence scores using gradient-supporting model"""
    print(f"\n🔍 Computing Influence Function scores...")
    
    # Use embeddings for influence computation (LogisticRegression needs embeddings)
    embeddings_fetcher = (
        DataFetcher("imdb-embeddings", cache_dir="../data_files/", force_download=False, random_state=42)
        .split_dataset_by_count(len(data_fetcher.y_train), len(data_fetcher.y_test), len(data_fetcher.y_test))
    )
    
    if hasattr(data_fetcher, 'noise_indices'):
        # Apply same noise pattern to embeddings
        embeddings_fetcher = embeddings_fetcher.noisify("mix_labels", noise_rate=0.3)
    
    # Create gradient-supporting model for influence computation
    input_dim = embeddings_fetcher.x_train.shape[1] if hasattr(embeddings_fetcher.x_train, 'shape') else 768
    influence_model = LogisticRegression(input_dim=input_dim, num_classes=2)
    
    # Create influence function evaluator
    evaluator = InfluenceFunction()
    
    start_time = time.time()
    
    # Train evaluator
    evaluator_instance = evaluator.train(embeddings_fetcher, influence_model, metric="accuracy")
    
    # Compute influence scores
    influence_scores = evaluator_instance.evaluate_data_values()
    
    computation_time = time.time() - start_time
    
    print(f"✅ Influence computation complete")
    print(f"   Computation time: {computation_time:.2f}s")
    print(f"   Score range: [{np.min(influence_scores):.2f}, {np.max(influence_scores):.2f}]")
    print(f"   Score mean: {np.mean(influence_scores):.3f}")
    print(f"   Score std: {np.std(influence_scores):.3f}")
    
    return influence_scores, computation_time

def select_high_influence_samples(data_fetcher, influence_scores, select_ratio=0.9):
    """Select top 90% high-influence samples"""
    print(f"\n🎯 Selecting high-influence samples...")
    print(f"   Selection ratio: {select_ratio*100:.0f}%")
    
    n_total = len(influence_scores)
    n_select = int(n_total * select_ratio)
    
    # Sort by influence score (descending - highest first)
    sorted_indices = np.argsort(influence_scores)[::-1]
    high_influence_indices = sorted_indices[:n_select]
    low_influence_indices = sorted_indices[n_select:]
    
    # Create filtered dataset
    filtered_x_train = data_fetcher.x_train[high_influence_indices]
    filtered_y_train = data_fetcher.y_train[high_influence_indices]
    
    high_scores = influence_scores[high_influence_indices]
    low_scores = influence_scores[low_influence_indices]
    
    print(f"✅ Sample selection complete")
    print(f"   Original samples: {n_total}")
    print(f"   Selected samples: {n_select} ({select_ratio*100:.0f}%)")
    print(f"   Rejected samples: {n_total - n_select} ({(1-select_ratio)*100:.0f}%)")
    print(f"   High-influence mean: {np.mean(high_scores):.3f}")
    print(f"   Low-influence mean: {np.mean(low_scores):.3f}")
    print(f"   Separation power: {np.mean(high_scores) - np.mean(low_scores):.3f}")
    
    return {
        "filtered_x_train": filtered_x_train,
        "filtered_y_train": filtered_y_train,
        "high_influence_indices": high_influence_indices,
        "low_influence_indices": low_influence_indices,
        "selection_stats": {
            "total_samples": n_total,
            "selected_samples": n_select,
            "selection_ratio": select_ratio,
            "high_influence_mean": np.mean(high_scores),
            "low_influence_mean": np.mean(low_scores),
            "separation_power": np.mean(high_scores) - np.mean(low_scores)
        }
    }

def train_cleaned_bert_model(filtered_data, data_fetcher, epochs=5):
    """Train BERT model on cleaned high-influence data"""
    print(f"\n🧹 Training Cleaned BERT Model (high-influence data only)")
    print(f"   Training samples: {len(filtered_data['filtered_y_train'])}")
    print(f"   Epochs: {epochs}")
    
    # Create new BERT model
    cleaned_bert_model = BertClassifier(
        pretrained_model_name="distilbert-base-uncased",
        num_classes=2
    )
    
    start_time = time.time()
    
    # Train on filtered high-influence data
    cleaned_bert_model.fit(
        filtered_data["filtered_x_train"], 
        filtered_data["filtered_y_train"],
        epochs=epochs,
        batch_size=16
    )
    
    training_time = time.time() - start_time
    
    # Evaluate cleaned model
    train_acc = calculate_accuracy(cleaned_bert_model, filtered_data["filtered_x_train"], filtered_data["filtered_y_train"])
    test_acc = calculate_accuracy(cleaned_bert_model, data_fetcher.x_test, data_fetcher.y_test)
    
    print(f"✅ Cleaned BERT training complete")
    print(f"   Training time: {training_time:.1f}s")
    print(f"   Training accuracy: {train_acc:.4f}")
    print(f"   Test accuracy: {test_acc:.4f}")
    
    return cleaned_bert_model, {
        "training_time": training_time,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc
    }

def create_comparison_plots(original_results, cleaned_results, influence_scores, selection_data, output_dir):
    """Create comparison visualization plots"""
    print(f"\n📊 Creating comparison plots...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("BERT Influence Function Retraining Results", fontsize=16, fontweight='bold')
    
    # Training accuracy comparison
    models = ['Original BERT\n(with noise)', 'Cleaned BERT\n(90% high-influence)']
    train_accs = [original_results['train_accuracy'], cleaned_results['train_accuracy']]
    test_accs = [original_results['test_accuracy'], cleaned_results['test_accuracy']]
    
    axes[0, 0].bar(models, train_accs, color=['red', 'green'], alpha=0.7, edgecolor='black')
    axes[0, 0].set_title("Training Accuracy Comparison")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(train_accs):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Test accuracy comparison
    axes[0, 1].bar(models, test_accs, color=['red', 'green'], alpha=0.7, edgecolor='black')
    axes[0, 1].set_title("Test Accuracy Comparison")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(test_accs):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Training time comparison
    train_times = [original_results['training_time'], cleaned_results['training_time']]
    axes[1, 0].bar(models, train_times, color=['red', 'green'], alpha=0.7, edgecolor='black')
    axes[1, 0].set_title("Training Time Comparison")
    axes[1, 0].set_ylabel("Time (seconds)")
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(train_times):
        axes[1, 0].text(i, v + v*0.05, f'{v:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # Influence score distribution with selection
    sample_indices = np.arange(len(influence_scores))
    high_indices = selection_data["high_influence_indices"]
    colors = ['green' if i in high_indices else 'red' for i in sample_indices]
    
    axes[1, 1].scatter(sample_indices, influence_scores, c=colors, alpha=0.7, s=30)
    axes[1, 1].axhline(0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].set_title("Influence Score Distribution\n(Green: Selected, Red: Rejected)")
    axes[1, 1].set_xlabel("Sample Index")
    axes[1, 1].set_ylabel("Influence Scores")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "bert_influence_retraining_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plots saved to: {output_path}")

def save_results(original_results, cleaned_results, influence_scores, selection_data, output_dir):
    """Save complete results to JSON"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = {
        "experiment_config": {
            "dataset": "imdb",
            "method": "influence_function",
            "selection_ratio": selection_data["selection_stats"]["selection_ratio"]
        },
        "original_model": original_results,
        "cleaned_model": cleaned_results,
        "influence_analysis": {
            "scores_stats": {
                "mean": float(np.mean(influence_scores)),
                "std": float(np.std(influence_scores)),
                "min": float(np.min(influence_scores)),
                "max": float(np.max(influence_scores))
            },
            "selection_stats": selection_data["selection_stats"]
        },
        "performance_improvement": {
            "test_accuracy_change": cleaned_results["test_accuracy"] - original_results["test_accuracy"],
            "training_time_change": cleaned_results["training_time"] - original_results["training_time"],
            "data_reduction": 1 - selection_data["selection_stats"]["selection_ratio"]
        }
    }
    
    with open(output_path / "bert_influence_retraining_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def print_final_summary(original_results, cleaned_results, selection_data):
    """Print final experiment summary"""
    print(f"\n🎉 BERT INFLUENCE FUNCTION RETRAINING COMPLETE")
    print("=" * 70)
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"   Original BERT (with noise):")
    print(f"     • Training accuracy: {original_results['train_accuracy']:.4f}")
    print(f"     • Test accuracy: {original_results['test_accuracy']:.4f}")
    print(f"     • Training time: {original_results['training_time']:.1f}s")
    
    print(f"\n   Cleaned BERT (90% high-influence):")
    print(f"     • Training accuracy: {cleaned_results['train_accuracy']:.4f}")
    print(f"     • Test accuracy: {cleaned_results['test_accuracy']:.4f}")
    print(f"     • Training time: {cleaned_results['training_time']:.1f}s")
    
    test_improvement = cleaned_results['test_accuracy'] - original_results['test_accuracy']
    time_change = cleaned_results['training_time'] - original_results['training_time']
    
    print(f"\n🎯 IMPROVEMENT ANALYSIS:")
    print(f"   • Test accuracy change: {test_improvement:+.4f}")
    print(f"   • Training time change: {time_change:+.1f}s")
    print(f"   • Data reduction: {(1-selection_data['selection_stats']['selection_ratio'])*100:.0f}%")
    print(f"   • Sample selection effectiveness: {selection_data['selection_stats']['separation_power']:.2f}")

def main():
    """Main pipeline execution"""
    args = parse_arguments()
    
    print("🚀 BERT Sentiment Analysis with Influence Function Retraining Pipeline")
    print("=" * 80)
    print(f"📋 Configuration:")
    print(f"   Training samples: {args.train_count}")
    print(f"   Test samples: {args.test_count}")
    print(f"   Noise rate: {args.noise_rate*100:.1f}%")
    print(f"   Selection ratio: {args.select_ratio*100:.0f}%")
    print(f"   Epochs: {args.epochs}")
    print(f"   Output directory: {args.output_dir}")
    
    start_time = time.time()
    
    # Step 1: Setup data with noise
    data_fetcher = setup_data_and_noise(args.train_count, args.test_count, args.noise_rate, args.seed)
    
    # Step 2: Train original BERT model
    original_bert, original_results = train_original_bert_model(data_fetcher, args.epochs)
    
    # Step 3: Compute influence scores
    influence_scores, computation_time = compute_influence_scores(data_fetcher)
    
    # Step 4: Select high-influence samples
    selection_data = select_high_influence_samples(data_fetcher, influence_scores, args.select_ratio)
    
    # Step 5: Train cleaned BERT model
    cleaned_bert, cleaned_results = train_cleaned_bert_model(selection_data, data_fetcher, args.epochs)
    
    # Step 6: Create comparison plots
    create_comparison_plots(original_results, cleaned_results, influence_scores, selection_data, args.output_dir)
    
    # Step 7: Save results
    save_results(original_results, cleaned_results, influence_scores, selection_data, args.output_dir)
    
    # Step 8: Print final summary
    print_final_summary(original_results, cleaned_results, selection_data)
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total pipeline time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"📁 All results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()