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
import numpy as np

def train_high_order_hmm(
    sequences,
    labels,
    states,
    alphabet,
    order=5
):
    """
    Train a supervised high-order HMM.

    Parameters
    ----------
    sequences : list[str]
        List of observation sequences (e.g. DNA strings)
    labels : list[str]
        Corresponding list of state label sequences
    states : list[str]
        List of possible states (e.g. ['N','S','1','2','3','E'])
    alphabet : list[str]
        Observation alphabet (e.g. ['A','C','G','T'])
    order : int
        Emission order (0 = standard HMM)

    Returns
    -------
    emit_prob : np.ndarray
        Shape (n_states, 4^order, |alphabet|)
    trans_prob : np.ndarray
        Shape (n_states, n_states)
    start_prob : np.ndarray
        Shape (n_states,)
    """

    # ----------------------
    # Index mappings
    # ----------------------
    state_to_idx = {s: i for i, s in enumerate(states)}
    sym_to_idx = {a: i for i, a in enumerate(alphabet)}

    n_states = len(states)
    n_symbols = len(alphabet)
    n_contexts = (n_symbols ** order) if order > 0 else 1

    # ----------------------
    # Initialize counts (Laplace smoothing)
    # ----------------------
    start_counts = np.ones(n_states)
    trans_counts = np.ones((n_states, n_states))
    emit_counts = np.ones((n_states, n_contexts, n_symbols))

    # ----------------------
    # Training loop
    # ----------------------
    for seq, lbl in zip(sequences, labels):
        if len(seq) != len(lbl):
            continue

        # Convert sequence to integer indices
        seq_ints = [sym_to_idx.get(c, 0) for c in seq]

        # Context handling
        current_ctx = 0
        if order > 0:
            mask = (1 << (2 * order)) - 1

        for t in range(len(seq)):
            s_char = lbl[t]
            if s_char not in state_to_idx:
                continue

            s_idx = state_to_idx[s_char]
            n_idx = seq_ints[t]

            # Start probability
            if t == 0:
                start_counts[s_idx] += 1

            # Emission count
            emit_counts[s_idx, current_ctx, n_idx] += 1

            # Transition count
            if t < len(seq) - 1:
                next_state = lbl[t + 1]
                if next_state in state_to_idx:
                    trans_counts[s_idx, state_to_idx[next_state]] += 1

            # Update context
            if order > 0:
                current_ctx = ((current_ctx << 2) | n_idx) & mask

    # ----------------------
    # Normalize to probabilities
    # ----------------------
    start_prob = start_counts / start_counts.sum()
    trans_prob = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    emit_prob = emit_counts / emit_counts.sum(axis=2, keepdims=True)

    return emit_prob, trans_prob, start_prob


# ======================
# 2. HIGH-ORDER VITERBI (JIT)
# ======================
@jit(nopython=True)
def fast_high_order_viterbi(seq_ints, emit_probs, trans_probs, start_probs, alphabet, states, order = 5):
    # 2. Log Probs
    eps = 1e-10
    start_log = np.log(start_probs + eps)
    trans_log = np.log(trans_probs + eps)
    emit_log_3d = np.log(emit_probs + eps) # Shape (6, 4^k, 4)
    
    n_obs = len(seq_ints)
    n_states = len(states)

    idx_N = states.index('N')
    idx_S = states.index('S')
    idx_1 = states.index('1')
    idx_2 = states.index('2')
    idx_3 = states.index('3')
    idx_E = states.index('E')  
      
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
    # s0, t0, e0 = get_train_data_high_order(order=0)
    # # e0 is 3D [6, 1, 4], flatten to [6, 4] for hmmlearn
    # e0_flat = e0[:, 0, :]
    
    # print("\n>>> Running Exp 1: NORMAL (hmmlearn, No Constraints)...")
    # clean_test_seq = "".join([test_seq[i] for i in valid_indices])
    # start_time = time.time()
    # pred_norm = run_normal_viterbi(clean_test_seq, s0, t0, e0_flat)
    # print(f"Done ({time.time()-start_time:.2f}s)")
    # results['Normal'] = print_detailed_stats(clean_true_labels, pred_norm, "Normal_Baseline")

    # ==========================
    # EXPERIMENT 2-5: CUSTOM ORDERS
    # ==========================
    # Orders to test
    ORDERS_TO_TEST = [5]



    
    for k in ORDERS_TO_TEST:
        print(f"\n>>> Running Exp: CUSTOM ORDER {k} (With Constraints)...")
        
        # 1. Train
        emmitions_p_3d, trans_prob, start_prob = get_train_data_high_order(order=k)
        
        # 2. Log Probs
        # eps = 1e-10
        # s_log = np.log(start_prob + eps)
        # t_log = np.log(trans_prob + eps)
        # e_log = np.log(emmitions_p_3d + eps) # Shape (6, 4^k, 4)
        
        # 3. Run Viterbi
        start_time = time.time()
        # Numba needs fixed types. e_log is 3D float array.
        pred_path = fast_high_order_viterbi(seq_ints, emmitions_p_3d, trans_prob, start_prob, order=k)
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