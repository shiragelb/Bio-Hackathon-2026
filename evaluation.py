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

# --- גרף 1: מטריצות בלבול (Grid) ---
def plot_confusion_matrix_grid(results_list):
    if not results_list: return
    df = pd.DataFrame(results_list)
    
    # שימוש בשם הארוך (Category) ככותרת
    categories = df['Category'].unique()
    n_cats = len(categories)
    
    # חישוב גודל הגריד
    cols = 2
    rows = math.ceil(n_cats / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    if n_cats == 1: axes = np.array([axes]) # טיפול במקרה של ניסוי יחיד
    axes = axes.flatten()
    
    for i, cat in enumerate(categories):
        subset = df[df['Category'] == cat]
        
        # סכימה של המדדים
        total_tp = subset['TP'].sum()
        total_fp = subset['FP'].sum()
        total_fn = subset['FN'].sum()
        total_tn = subset['TN'].sum()
        
        cm = np.array([[total_tn, total_fp], [total_fn, total_tp]])
        # נרמול לאחוזים
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm) # הגנה מפני חלוקה באפס

        sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', cbar=False,
                    xticklabels=['Pred: N', 'Pred: C'],
                    yticklabels=['True: N', 'True: C'],
                    ax=axes[i])
        
        # הקטנת הפונט של הכותרת כדי שהשם הארוך ייכנס
        axes[i].set_title(cat, fontsize=10, fontweight='bold')
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.show()

# --- גרף 2: השוואה קבוצתית (חכם) ---
def plot_grouped_benchmark(results_list, metric='F1_Score'):
    """
    מפרק אוטומטית את השם 'Train: X | Test: Y' כדי ליצור גרף מקובץ יפה.
    """
    if not results_list: return
    df = pd.DataFrame(results_list)
    
    # לוגיקה חכמה: ננסה לפרק את השם הארוך ל-Train ו-Test באופן זמני רק לגרף הזה
    def parse_category(cat):
        if '|' in cat:
            parts = cat.split('|')
            # Train יהיה ציר ה-X, Test יהיה הצבע (Hue)
            return parts[0].strip(), parts[1].strip()
        return cat, "General"

    # יצירת עמודות זמניות
    df[['Train_Group', 'Test_Group']] = df['Category'].apply(
        lambda x: pd.Series(parse_category(x))
    )
    
    plt.figure(figsize=(14, 8))
    
    sns.barplot(
        data=df, 
        x='Train_Group', 
        y=metric, 
        hue='Test_Group',
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
    plt.show()

# --- גרף 3: התפלגות (Violin) ---
def plot_violin_distributions(results_list, metric='F1_Score'):
    if not results_list: return
    df = pd.DataFrame(results_list)
    
    plt.figure(figsize=(12, 8)) # הגדלנו קצת כדי שיהיה מקום לשמות
    sns.violinplot(x='Category', y=metric, data=df, inner="stick", palette="Pastel1", density_norm='width')
    
    plt.title(f'Stability Distribution ({metric})', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10) # סיבוב חד לשמות ארוכים
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()

# --- הוספה לסוף הקובץ evaluation.py ---

def plot_tuning_curve(tuning_results):
    """
    מצייר גרף של F1 Score כתלות בערך ה-Penalty.
    """
    if not tuning_results: return
    df = pd.DataFrame(tuning_results)
    
    plt.figure(figsize=(10, 6))
    
    # גרף קו: ציר X הוא עוצמת השינוי, ציר Y הוא ה-F1
    # hue='Genome' מאפשר לראות אם השינוי משפיע אותו דבר על כל הגנומים
    sns.lineplot(data=df, x='Penalty', y='F1_Score', hue='Genome', marker='o', palette='viridis')
    
    plt.title('Hyperparameter Tuning: Effect of Transition Penalty on F1', fontsize=16)
    plt.xlabel('Transition Penalty (Reduction in P(C->C))')
    plt.ylabel('F1 Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()