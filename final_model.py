import numpy as np
import pandas as pd
from pathlib import Path
from hmmlearn import hmm
import time
import sys
from numba import jit
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# Config & Constants
# ======================
BASE_DIR = Path("processed_data_2")

STATES = ['N', 'S', '1', '2', '3', 'E']
NUCLEOTIDES = ['A', 'C', 'G', 'T']
STOP_CODONS = ["TAA", "TAG", "TGA"]

state_to_idx = {s: i for i, s in enumerate(STATES)}
nuc_to_idx = {n: i for i, n in enumerate(NUCLEOTIDES)}

# The file we want to TEST on
TEST_GENOME_NAME = "Escherichia_coli_042_genome.fasta"
TEST_LABEL_NAME = "Escherichia_coli_042_labels.fasta"

# ======================
# Helper: Read Files
# ======================
def read_fasta_seq(filepath):
    path = BASE_DIR / filepath
    with open(path, "r") as f:
        _ = f.readline() 
        seq = f.readline().strip()
    return seq

# ======================
# 1. ADVANCED TRAINING (Variable Order)
# ======================
def get_train_data_high_order(order=0):
    """
    Trains an emission model of order 'k'.
    Order 0: P(Nuc | State) (Standard)
    Order 5: P(Nuc | Prev 5 Nucs, State)
    """
    all_files = list(BASE_DIR.glob("*_genome.fasta"))
    TRAIN_FILES = []
    
    for g in all_files:
        if g.name == TEST_GENOME_NAME: continue
        l_name = g.name.replace("_genome.fasta", "_labels.fasta")
        if (BASE_DIR / l_name).exists():
            TRAIN_FILES.append((g.name, l_name))

    print(f"Training Order-{order} Model on {len(TRAIN_FILES)} genomes...")
    
    n_states = len(STATES)
    n_symbols = len(NUCLEOTIDES)
    
    # Context Size: 4^order (e.g., Order 2 -> 4^2 = 16 contexts)
    n_contexts = 4 ** order
    
    # Matrices
    # Trans and Start are always 0th order (State -> State)
    trans_counts = np.ones((n_states, n_states))
    start_counts = np.ones(n_states)
    
    # Emission: [State, Context_Index, Next_Nucleotide]
    # We use ones() for Laplace Smoothing (so we never have 0 probability)
    emit_counts = np.ones((n_states, n_contexts, n_symbols))
    
    for g_file, l_file in TRAIN_FILES:
        g_seq = read_fasta_seq(g_file)
        l_seq = read_fasta_seq(l_file)
        if len(g_seq) != len(l_seq): continue
        
        # We process the sequence
        # We need to maintain a "rolling context"
        
        # Convert entire sequence to ints for speed
        seq_ints = [nuc_to_idx.get(c, 0) for c in g_seq] # Default to A if N
        
        # Mask to keep only the last 'order' bits (for bitwise shift)
        # e.g. for Order 2 (4 bits), mask is 1111 binary = 15
        mask = (1 << (2 * order)) - 1
        
        current_ctx = 0 # Start with context 0 (AAAA...)
        
        for t in range(len(g_seq)):
            s_idx = state_to_idx.get(l_seq[t])
            n_idx = seq_ints[t]
            
            if s_idx is None: continue
            
            # Update counts
            if t == 0: start_counts[s_idx] += 1
            
            # Emission Count: State + Context -> Nucleotide
            emit_counts[s_idx, current_ctx, n_idx] += 1
            
            # Transition Count
            if t < len(g_seq) - 1:
                next_s = state_to_idx.get(l_seq[t+1])
                if next_s is not None: trans_counts[s_idx, next_s] += 1
            
            # Update Context (Shift left, add new nucleotide, mask)
            # This effectively slides the window by 1
            if order > 0:
                current_ctx = ((current_ctx << 2) | n_idx) & mask

    # Normalize
    start_prob = start_counts / start_counts.sum()
    trans_prob = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    # Normalize emission along the last axis (sum of A,C,G,T must be 1)
    emit_prob = emit_counts / emit_counts.sum(axis=2, keepdims=True)
    
    return start_prob, trans_prob, emit_prob

# ======================
# 2. HIGH-ORDER VITERBI (JIT)
# ======================
@jit(nopython=True)
def fast_high_order_viterbi(seq_ints, start_log, trans_log, emit_log_3d, order):
    n_obs = len(seq_ints)
    n_states = 6
    idx_N, idx_S, idx_1, idx_2, idx_3, idx_E = 0, 1, 2, 3, 4, 5
    
    viterbi = np.full((n_states, n_obs), -np.inf)
    backpointer = np.zeros((n_states, n_obs), dtype=np.int32)
    
    # Initial Context (AAAA...)
    current_ctx = 0
    mask = (1 << (2 * order)) - 1
    
    # Initialize t=0
    first_nuc = seq_ints[0]
    # For t=0, context is 0. 
    viterbi[:, 0] = start_log + emit_log_3d[:, 0, first_nuc]
    
    # Update context for next step
    if order > 0:
        current_ctx = ((current_ctx << 2) | first_nuc) & mask

    # Main Loop
    for t in range(1, n_obs):
        nuc_idx = seq_ints[t]
        
        # --- LOOK AHEAD for Constraints ---
        is_atg = False
        is_stop = False
        if t + 2 < n_obs:
            n1, n2, n3 = seq_ints[t], seq_ints[t+1], seq_ints[t+2]
            # ATG (0,3,2)
            if n1==0 and n2==3 and n3==2: is_atg = True
            # Stops: TAA(300), TAG(302), TGA(320)
            if n1==3:
                if (n2==0 and n3==0) or (n2==0 and n3==2) or (n2==2 and n3==0):
                    is_stop = True

        for curr_s in range(n_states):
            best_score = -np.inf
            best_prev = -1
            
            for prev_s in range(n_states):
                score = viterbi[prev_s, t-1] + trans_log[prev_s, curr_s]
                
                # --- CONSTRAINTS ---
                # 1. Start (N->S needs ATG)
                if curr_s == idx_S and prev_s == idx_N:
                    if not is_atg: score = -np.inf
                # 2. Stop (3->E needs Stop)
                if curr_s == idx_E and prev_s == idx_3:
                    if not is_stop: score = -np.inf
                # 3. Steamroller (3->1 forbidden at Stop)
                if curr_s == idx_1 and prev_s == idx_3:
                    if is_stop: score = -np.inf
                
                if score > best_score:
                    best_score = score
                    best_prev = prev_s
            
            # --- HIGH ORDER EMISSION LOOKUP ---
            # We use 'current_ctx' which represents prev nucleotides
            emission = emit_log_3d[curr_s, current_ctx, nuc_idx]
            
            viterbi[curr_s, t] = best_score + emission
            backpointer[curr_s, t] = best_prev

        # Update Context for next step 't+1'
        if order > 0:
            current_ctx = ((current_ctx << 2) | nuc_idx) & mask

    # Backtrack
    path = np.zeros(n_obs, dtype=np.int32)
    path[-1] = np.argmax(viterbi[:, -1])
    for t in range(n_obs-2, -1, -1):
        path[t] = backpointer[path[t+1], t+1]
    
    return path

# ======================
# STATS & PLOTTING
# ======================
def print_detailed_stats(true_labels, pred_indices, title="Model"):
    print(f"\n{'='*20} {title} STATISTICS {'='*20}")
    
    # Filter only valid chars (remove skipped ones)
    valid = min(len(true_labels), len(pred_indices))
    t_lbl = true_labels[:valid]
    p_idx = pred_indices[:valid]
    
    # Accuracy
    correct = sum(1 for t, p in zip(t_lbl, p_idx) if t == STATES[p])
    acc = (correct / valid) * 100
    
    # Global Counts
    counts = pd.DataFrame({
        'Actual': pd.Series(list(t_lbl)).value_counts(),
        'Pred': pd.Series([STATES[i] for i in p_idx]).value_counts()
    }).fillna(0).astype(int)
    
    print(f"OVERALL ACCURACY: {acc:.2f}%")
    print("\n--- Volume Check ---")
    print(counts)
    
    # Save Matrix
    conf_matrix = pd.crosstab(
        pd.Series(list(t_lbl), name='Actual'),
        pd.Series([STATES[i] for i in p_idx], name='Predicted')
    )
    conf_pct = conf_matrix.div(conf_matrix.sum(axis=1), axis=0) * 100
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_pct, annot=True, fmt='.1f', cmap='Blues')
    plt.title(f'{title} Accuracy: {acc:.2f}%')
    plt.savefig(f"confusion_{title.split()[0]}.png")
    plt.close()
    
    return acc

# ======================
# ALGORITHM 1: NORMAL (Baseline)
# ======================
def run_normal_viterbi(seq, start_p, trans_p, emit_p):
    # Flatten emit_p if it's high order (Normal hmmlearn only supports Order 0)
    # We take the mean or just use Order 0 training for this one
    model = hmm.CategoricalHMM(n_components=len(STATES))
    model.startprob_ = start_p
    model.transmat_ = trans_p
    model.emissionprob_ = emit_p # Expecting 2D here
    X = [[nuc_to_idx[c]] for c in seq if c in nuc_to_idx]
    _, preds = model.decode(np.array(X))
    return preds

# ======================
# MAIN EXPERIMENT LOOP
# ======================
if __name__ == "__main__":
    # 1. Load Test Data
    print(f"Reading Test File: {TEST_GENOME_NAME}")
    test_seq = read_fasta_seq(TEST_GENOME_NAME)
    true_labels = read_fasta_seq(TEST_LABEL_NAME)
    
    # Convert Test Seq to Ints (Once)
    seq_ints = []
    valid_indices = []
    for i, char in enumerate(test_seq):
        if char in nuc_to_idx:
            seq_ints.append(nuc_to_idx[char])
            valid_indices.append(i)
    seq_ints = np.array(seq_ints, dtype=np.int32)
    
    clean_true_labels = "".join([true_labels[i] for i in valid_indices])
    
    results = {}

    # ==========================
    # EXPERIMENT 1: NORMAL (Baseline)
    # ==========================
    # Train Order 0 for Normal
    s0, t0, e0 = get_train_data_high_order(order=0)
    # e0 is 3D [6, 1, 4], flatten to [6, 4] for hmmlearn
    e0_flat = e0[:, 0, :]
    
    print("\n>>> Running Exp 1: NORMAL (hmmlearn, No Constraints)...")
    clean_test_seq = "".join([test_seq[i] for i in valid_indices])
    start_time = time.time()
    pred_norm = run_normal_viterbi(clean_test_seq, s0, t0, e0_flat)
    print(f"Done ({time.time()-start_time:.2f}s)")
    results['Normal'] = print_detailed_stats(clean_true_labels, pred_norm, "Normal_Baseline")

    # ==========================
    # EXPERIMENT 2-5: CUSTOM ORDERS
    # ==========================
    # Orders to test
    ORDERS_TO_TEST = [0, 3, 4, 5]
    
    for k in ORDERS_TO_TEST:
        print(f"\n>>> Running Exp: CUSTOM ORDER {k} (With Constraints)...")
        
        # 1. Train
        s_p, t_p, e_p_3d = get_train_data_high_order(order=k)
        
        # 2. Log Probs
        eps = 1e-10
        s_log = np.log(s_p + eps)
        t_log = np.log(t_p + eps)
        e_log = np.log(e_p_3d + eps) # Shape (6, 4^k, 4)
        
        # 3. Run Viterbi
        start_time = time.time()
        # Numba needs fixed types. e_log is 3D float array.
        pred_path = fast_high_order_viterbi(seq_ints, s_log, t_log, e_log, k)
        print(f"Done ({time.time()-start_time:.2f}s)")
        
        # 4. Stats
        acc = print_detailed_stats(clean_true_labels, pred_path, f"Custom_Order_{k}")
        results[f"Order_{k}"] = acc

    # ==========================
    # FINAL SUMMARY
    # ==========================
    print("\n" + "="*40)
    print(" FINAL ACCURACY SUMMARY ")
    print("="*40)
    for name, acc in results.items():
        print(f"{name:<15}: {acc:.2f}%")