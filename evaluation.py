import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import math

sns.set_theme(style="whitegrid")

def convert_to_binary(seq_list):
    return np.array([1 if x == 'C' else 0 for x in seq_list])

def get_metrics_dict(y_true_raw, y_pred_raw, genome_name="Unknown"):
    y_true = convert_to_binary(y_true_raw)
    y_pred = convert_to_binary(y_pred_raw)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return {
        'Genome': genome_name,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'IoU': iou,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn
    }

def plot_confusion_matrix_grid(results_list, save_path):
    if not results_list: return
    df = pd.DataFrame(results_list)
    
    categories = df['Experiment'].unique()
    n_cats = len(categories)
    
    cols = 2
    rows = math.ceil(n_cats / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    if n_cats == 1: axes = np.array([axes]) 
    axes = axes.flatten()
    
    for i, cat in enumerate(categories):
        subset = df[df['Experiment'] == cat]
        
        total_tp = subset['TP'].sum()
        total_fp = subset['FP'].sum()
        total_fn = subset['FN'].sum()
        total_tn = subset['TN'].sum()
        
        cm = np.array([[total_tn, total_fp], [total_fn, total_tp]])
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm) 

        sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', cbar=False,
                    xticklabels=['Pred: N', 'Pred: C'],
                    yticklabels=['True: N', 'True: C'],
                    vmin=0,
                    vmax=1,
                    ax=axes[i])
        
        axes[i].set_title(cat, fontsize=10, fontweight='bold')
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(f"{save_path}/confusion_matrices.png")

def plot_grouped_benchmark(results_list, metric='F1_Score', save_path=None):

    if not results_list: return
    df = pd.DataFrame(results_list)
    
    # def parse_category(cat):
    #     if '|' in cat:
    #         parts = cat.split('|')
    #         # Train יהיה ציר ה-X, Test יהיה הצבע (Hue)
    #         return parts[0].strip(), parts[1].strip()
    #     return cat, "General"

    # # יצירת עמודות זמניות
    # df[['Train_Group', 'Test_Group']] = df['Category'].apply(
    #     lambda x: pd.Series(parse_category(x))
    # )
    
    plt.figure(figsize=(14, 8))
    
    sns.barplot(
        data=df, 
        x='Train', 
        y=metric, 
        hue='Test',
        palette="viridis", 
        capsize=.05,
        edgecolor=".2"
    )
    
    plt.title(f'{metric} Analysis: Training Source vs Test Target', fontsize=16)
    plt.ylabel(metric)
    plt.xlabel("Training Set")
    plt.xticks(rotation=15, ha='right') # סיבוב קל לשמות הארוכים
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Test Condition")
    plt.tight_layout()
    plt.savefig(f"{save_path}/{metric}_grouped_benchmark.png")


def get_segments(y_binary):
    segments = []
    if len(y_binary) == 0: return segments
    curr_val = y_binary[0]
    start = 0
    for i in range(1, len(y_binary)):
        if y_binary[i] != curr_val:
            segments.append((start, i, curr_val))
            curr_val = y_binary[i]
            start = i
    segments.append((start, len(y_binary), curr_val))
    return segments

def collect_gene_length_stats(y_true_raw, y_pred_raw):

    y_true = np.array([1 if x == 'C' else 0 for x in y_true_raw]) if isinstance(y_true_raw[0], str) else y_true_raw
    y_pred = np.array([1 if x == 'C' else 0 for x in y_pred_raw]) if isinstance(y_pred_raw[0], str) else y_pred_raw

    true_genes = [s for s in get_segments(y_true) if s[2] == 1]
    
    stats = []
    
    for start, end, _ in true_genes:
        length = end - start
        
        pred_segment = y_pred[start:end]
        
        gene_coverage = np.mean(pred_segment)
        
        stats.append({
            'Length': length,
            'Coverage': gene_coverage 
        })
        
    return stats

def plot_success_by_length_binned(all_length_data, save_path):

    if not all_length_data: return

    df = pd.DataFrame(all_length_data)
    
    bins = [0, 50, 100, 200, 300, 450, 600, 1000, 1500, 3000, 10000]
    labels = ['0-50', '50-100', '100-200', '200-300', '300-450', '450-600', '600-1k', '1k-1.5k', '1.5k-3k', '>3k']
    
    df['Length_Bin'] = pd.cut(df['Length'], bins=bins, labels=labels)
    
    plt.figure(figsize=(12, 7))
    
    # ציור הבר-פלוט
    sns.barplot(
        data=df, 
        x='Length_Bin', 
        y='Coverage', 
        hue='Experiment',   
        palette="magma",
        errorbar=('ci', 95),  
        capsize=0.1
    )
    
    plt.title('Prediction Success by Gene Length', fontsize=18)
    plt.xlabel('Gene Length (bp)', fontsize=12)
    plt.ylabel('Average Coverage (Recall)', fontsize=12)
    plt.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/gene_length_success.png")

# --- פונקציה 3: איסוף נתונים לאזורים לא-מקודדים ---
def collect_non_coding_length_stats(y_true_raw, y_pred_raw):
    y_true = np.array([1 if x == 'C' else 0 for x in y_true_raw]) if isinstance(y_true_raw[0], str) else y_true_raw
    y_pred = np.array([1 if x == 'C' else 0 for x in y_pred_raw]) if isinstance(y_pred_raw[0], str) else y_pred_raw

    non_coding_segments = [s for s in get_segments(y_true) if s[2] == 0]
    
    stats = []
    
    for start, end, _ in non_coding_segments:
        length = end - start
        
        pred_segment = y_pred[start:end]
        
        success_rate = np.mean(1 - pred_segment)
        
        stats.append({
            'Length': length,
            'Coverage': success_rate 
        })
        
    return stats

def plot_non_coding_success_by_length(all_length_data, save_path):
    if not all_length_data: return

    df = pd.DataFrame(all_length_data)
    
    bins = [0, 50, 100, 200, 300, 450, 600, 1000, 2000, 5000]
    labels = ['0-50', '50-100', '100-200', '200-300', '300-450', '450-600', '600-1k', '1k-2k', '>2k']
    
    df['Length_Bin'] = pd.cut(df['Length'], bins=bins, labels=labels)
    
    plt.figure(figsize=(12, 7))
    
    sns.barplot(
        data=df, 
        x='Length_Bin', 
        y='Coverage', 
        hue='Experiment',
        palette="coolwarm", 
        errorbar=('ci', 95),
        capsize=0.1
    )
    
    plt.title('Prediction Success by Non-Coding Region Length', fontsize=18)
    plt.xlabel('Non-Coding Region Length (bp)', fontsize=12)
    plt.ylabel('Average Success (Prediction of N)', fontsize=12)
    plt.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/non_coding_length_success.png")