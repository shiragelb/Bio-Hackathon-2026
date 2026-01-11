import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

# Setting the style for plots
sns.set_theme(style="whitegrid")

def convert_to_binary(seq_list):
    """Converts a list of C/N to 1/0 for mathematical calculations"""
    return np.array([1 if x == 'C' else 0 for x in seq_list])

def get_metrics_dict(y_true_raw, y_pred_raw, genome_name="Unknown"):
    """
    Computes statistical metrics and returns a dictionary of results.
    """
    # Convert to binary
    y_true = convert_to_binary(y_true_raw)
    y_pred = convert_to_binary(y_pred_raw)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    # Compute Jaccard Index (IoU)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return {
        'Genome': genome_name,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'IoU': iou
    }

def plot_confusion_matrix(y_true_raw, y_pred_raw, title="Confusion Matrix"):
    """Displays the confusion matrix"""
    y_true = convert_to_binary(y_true_raw)
    y_pred = convert_to_binary(y_pred_raw)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Coding (N)', 'Coding (C)'], 
                yticklabels=['Non-Coding (N)', 'Coding (C)'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_genome_segment(y_true_raw, y_pred_raw, start=0, end=1000, title="Segment Visualization"):
    """Visually displays the prediction against the ground truth in a specific segment"""
    y_true = convert_to_binary(y_true_raw)
    y_pred = convert_to_binary(y_pred_raw)
    
    segment_true = y_true[start:end]
    segment_pred = y_pred[start:end]
    x = range(start, end)
    
    fig, ax = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    
    # Ground Truth plot
    ax[0].fill_between(x, segment_true, step="pre", alpha=0.6, color='green')
    ax[0].set_ylabel("Ground Truth", fontsize=12)
    ax[0].set_yticks([0, 1])
    ax[0].set_yticklabels(['N', 'C'])
    ax[0].set_title(f"{title} (bp {start}-{end})", fontsize=14)
    
    # Prediction plot   
    ax[1].fill_between(x, segment_pred, step="pre", alpha=0.6, color='blue')
    ax[1].set_ylabel("HMM Prediction", fontsize=12)
    ax[1].set_yticks([0, 1])
    ax[1].set_yticklabels(['N', 'C'])
    ax[1].set_xlabel("Position (bp)", fontsize=12)
    
    # Marking errors in red
    diff = segment_true != segment_pred
    if np.any(diff):
        ax[1].fill_between(x, 0, 1, where=diff, color='red', alpha=0.3, label='Error')
        ax[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_all_experiments(results_list, metric='F1_Score'):
    """
    Renders a bar plot comparing a specified metric across different experiments.
    results_list: list of dictionaries containing metrics and 'Category'
    """
    if not results_list:
        print("No results to plot.")
        return

    df = pd.DataFrame(results_list)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Category', y=metric, data=df, palette="viridis", capsize=.1)
    # Adding swarmplot for individual points
    sns.swarmplot(x='Category', y=metric, data=df, color="white", alpha=0.5)
    
    plt.title(f"Comparison: {metric} across Experiments", fontsize=16)
    plt.ylabel(metric)
    plt.xlabel("Experiment Type")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()