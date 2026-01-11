import numpy as np
import pandas as pd
from pathlib import Path
import time

# ======================
# Config & Constants
# ======================
BASE_DIR = Path("processed_data_2")

# שמות המצבים במודל החדש
STATES = ['N', 'S', '1', '2', '3', 'E']
NUCLEOTIDES = ['A', 'C', 'G', 'T']

state_to_idx = {s: i for i, s in enumerate(STATES)}
nuc_to_idx = {n: i for i, n in enumerate(NUCLEOTIDES)}

# רשימת הקבצים לאימון
TRAIN_FILES = [
    ("Escherichia_coli_K12-MG1655_genome.fasta", "Escherichia_coli_K12-MG1655_labels.fasta"),
    ("Escherichia_coli_E. coli B REL606_genome.fasta", "Escherichia_coli_E. coli B REL606_labels.fasta"),
    ("Escherichia_coli_HS_genome.fasta", "Escherichia_coli_HS_labels.fasta"),
    ("Escherichia_coli_SE11_genome.fasta", "Escherichia_coli_SE11_labels.fasta"),
]

# קובץ הבדיקה
TEST_FILES = (
    "Escherichia_coli_042_genome.fasta",
    "Escherichia_coli_042_labels.fasta"
)

# ======================
# 1. Helper Functions
# ======================
def read_fasta_file(path):
    """Reads a FASTA file and returns the sequence."""
    with open(path, "r") as f:
        lines = f.readlines()
    # מדלגים על השורה הראשונה (header) וקוראים את השאר
    return "".join(line.strip().upper() for line in lines[1:])

def train_hmm(file_pairs, base_dir):
    """
    מאמן HMM. 
    תיקון חשוב: לא מחליק מעברים (Transitions) כדי לשמור על המבנה הלוגי!
    """
    print(f"Training on {len(file_pairs)} genomes...")
    
    # אתחול מוונים
    transition_counts = np.zeros((len(STATES), len(STATES)))
    emission_counts = np.zeros((len(STATES), len(NUCLEOTIDES)))
    start_counts = np.zeros(len(STATES))

    for genome_fname, label_fname in file_pairs:
        g_path = base_dir / genome_fname
        l_path = base_dir / label_fname
        
        if not g_path.exists() or not l_path.exists():
            print(f"Warning: Missing file {g_path} or {l_path}, skipping.")
            continue

        print(f"  Processing {genome_fname}...")
        genome_seq = read_fasta_file(g_path)
        labels_seq = read_fasta_file(l_path)

        if len(genome_seq) != len(labels_seq):
            print(f"Error: Length mismatch in {genome_fname}. Skipping.")
            continue

        # המרה לאינדקסים (Optimized)
        g_idxs = np.array([nuc_to_idx.get(n, -1) for n in genome_seq])
        l_idxs = np.array([state_to_idx.get(l, -1) for l in labels_seq])

        # סינון תווים לא חוקיים
        valid_mask = (g_idxs != -1) & (l_idxs != -1)
        g_idxs = g_idxs[valid_mask]
        l_idxs = l_idxs[valid_mask]

        # 1. Start Counts
        if len(l_idxs) > 0:
            start_counts[l_idxs[0]] += 1

        # 2. Transitions (Vectorized)
        from_states = l_idxs[:-1]
        to_states = l_idxs[1:]
        np.add.at(transition_counts, (from_states, to_states), 1)

        # 3. Emissions
        np.add.at(emission_counts, (l_idxs, g_idxs), 1)

    # === נרמול (Smoothing Logic) ===
    
    # 1. Emissions: מוסיפים 1 כדי לאפשר פליטות נדירות (מוטציות)
    emission_counts += 1
    
    # 2. Transitions: לא נוגעים! (משאירים אפסים כאפסים)
    # זה התיקון הקריטי שמונע מהמודל להזות מעברים אסורים
    
    # 3. Start: מחליקים קצת
    start_counts += 1

    # חישוב הסתברויות
    emit_probs = emission_counts / emission_counts.sum(axis=1, keepdims=True)
    
    # חישוב זהיר למטריצת המעברים (טיפול בחלוקה באפס)
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    # איפה שהסכום הוא 0 (מצב שלא קרה), ההסתברות תהיה 0
    trans_probs = np.zeros_like(transition_counts)
    np.divide(transition_counts, row_sums, out=trans_probs, where=row_sums!=0)
    
    start_probs = start_counts / start_counts.sum()

    return trans_probs, emit_probs, start_probs

def viterbi_log(obs_seq, trans_probs, emit_probs, start_probs):
    """
    מימוש יעיל של ויטרבי במרחב הלוגריתמי.
    """
    n_states = trans_probs.shape[0]
    n_obs = len(obs_seq)
    
    # המרה ל-Log Probabilities
    # הוספת אפסילון קטן מאוד כדי ש-log(0) יהיה מספר שלילי ענק (עונש) ולא שגיאה
    eps = 1e-50
    log_trans = np.log(trans_probs + eps)
    log_emit = np.log(emit_probs + eps)
    log_start = np.log(start_probs + eps)

    # המרת הרצף למספרים
    obs_idxs = [nuc_to_idx.get(n, 0) for n in obs_seq] # ברירת מחדל A

    # מטריצות ויטרבי
    V = np.zeros((n_obs, n_states))
    B = np.zeros((n_obs, n_states), dtype=int)

    # אתחול (זמן 0)
    curr_emit = log_emit[:, obs_idxs[0]]
    V[0] = log_start + curr_emit

    # רקורסיה
    print(f"Running Viterbi on {n_obs} nucleotides...")
    
    for t in range(1, n_obs):
        if t % 500000 == 0:
            print(f"  ... step {t}/{n_obs}")
        
        # Broadcasting להוספת המעברים לערכים הקודמים
        scores = V[t-1][:, None] + log_trans
        
        # מציאת המקסימום
        best_prev = np.argmax(scores, axis=0)
        max_scores = scores[best_prev, np.arange(n_states)]
        
        # הוספת הפליטה הנוכחית
        V[t] = max_scores + log_emit[:, obs_idxs[t]]
        B[t] = best_prev

    # Traceback
    path = np.zeros(n_obs, dtype=int)
    path[-1] = np.argmax(V[-1])
    
    for t in range(n_obs - 2, -1, -1):
        path[t] = B[t+1, path[t+1]]

    return [STATES[i] for i in path]

def calculate_metrics(y_true, y_pred):
    """חישוב דיוק כללי ודיוק פר-מצב"""
    if len(y_true) != len(y_pred):
        print("Warning: Length mismatch in metrics calculation")
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    y_true = np.array(list(y_true))
    y_pred = np.array(y_pred)
    
    total = len(y_true)
    correct = np.sum(y_true == y_pred)
    accuracy = correct / total
    
    print(f"\nTotal Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    for s in STATES:
        true_pos = np.sum((y_true == s) & (y_pred == s))
        predicted_pos = np.sum(y_pred == s)
        actual_pos = np.sum(y_true == s)
        
        prec = true_pos / predicted_pos if predicted_pos > 0 else 0
        rec = true_pos / actual_pos if actual_pos > 0 else 0
        
        print(f"State {s}: Precision={prec:.3f}, Recall={rec:.3f} (Count: {actual_pos})")

# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    
    # 1. אימון
    print("\n=== Phase 1: Training ===")
    t_mat, e_mat, pi_vec = train_hmm(TRAIN_FILES, BASE_DIR)
    
    print("\nLearned Transition Matrix (Rounded):")
    print(pd.DataFrame(t_mat, index=STATES, columns=STATES).round(3))
    
    # 2. בדיקה
    print(f"\n=== Phase 2: Testing on {TEST_FILES[0]} ===")
    
    test_g_path = BASE_DIR / TEST_FILES[0]
    test_l_path = BASE_DIR / TEST_FILES[1]
    
    if test_g_path.exists():
        test_seq = read_fasta_file(test_g_path)
        true_labels = read_fasta_file(test_l_path)
        
        # === הערה: ===
        # השורה למטה מקצרת את הריצה ל-100,000 התווים הראשונים בלבד לצורך בדיקה מהירה.
        # כדי להריץ על כל הגנום (ייקח כמה דקות), שימי את השורות האלו בהערה (#).
        test_seq = test_seq[:100000]
        true_labels = true_labels[:100000]
        
        # הרצת ויטרבי
        start_time = time.time()
        predicted_labels = viterbi_log(test_seq, t_mat, e_mat, pi_vec)
        end_time = time.time()
        
        print(f"Viterbi completed in {end_time - start_time:.2f} seconds.")
        
        # 3. הערכה
        calculate_metrics(true_labels, predicted_labels)
        
    else:
        print(f"Test file not found: {test_g_path}")