import numpy as np
import pandas as pd
from pathlib import Path

import final_model  # משתמשים ב-fast_high_order_viterbi (JIT) + המילונים
import evaluation as ev
from experiments_config import EXPERIMENTS_CONFIG

# משתמשים בקוראי FASTA/LABELS ה"כלליים" (כמו run_experiments) כדי לא להניח FASTA בשורה אחת
from hmm import read_fasta, read_labels


# ======================
# Config
# ======================
MODEL_NAME = "final_model"
BASE_PATH = "preprocessed_5_hidden_states"

# מודל 6 מצבים (כמו final_model)
STATES = ['N', 'S', '1', '2', '3', 'E']
ALPHABET = ['A', 'C', 'G', 'T']

# ב-final_model יש STATE/NUC mappings גלובליים; נוודא שהם תואמים
state_to_idx = {s: i for i, s in enumerate(STATES)}
nuc_to_idx = {n: i for i, n in enumerate(ALPHABET)}

# איזה order זה "final" אצלך? (בדרך כלל 5)
FINAL_ORDER = 5

PLOTS_BASE = Path("plots") / MODEL_NAME
PLOTS_BASE.mkdir(parents=True, exist_ok=True)


# ======================
# Helpers: map 6-state -> binary C/N (כדי ש-evaluation.py יעבוד)
# ======================
def to_binary_cn(seq_states):
    """
    evaluation.py עובד על 'C'/'N'. לכן נמפה:
    N,E -> N (לא מקודד)
    S,1,2,3 -> C (מקודד)
    """
    coding = {'S', '1', '2', '3'}
    out = []
    for x in seq_states:
        out.append('C' if x in coding else 'N')
    return out


# ======================
# Train (כמו final_model.get_train_data_high_order) אבל על strain list מהקונפיג
# ======================
def train_high_order_from_strains(train_strains, order):
    """
    מחזיר:
      start_prob: (n_states,)
      trans_prob: (n_states, n_states)
      emit_prob:  (n_states, 4^order, 4)
    """
    n_states = len(STATES)
    n_symbols = len(ALPHABET)
    n_contexts = 4 ** order

    # Laplace smoothing
    trans_counts = np.ones((n_states, n_states), dtype=np.float64)
    start_counts = np.ones(n_states, dtype=np.float64)
    emit_counts = np.ones((n_states, n_contexts, n_symbols), dtype=np.float64)

    mask = (1 << (2 * order)) - 1  # same as final_model

    for strain in train_strains:
        genome_file = f"{BASE_PATH}/{strain}_genome.fasta"
        label_file = f"{BASE_PATH}/{strain}_labels.fasta"

        g_seq = read_fasta(genome_file, alphabet=ALPHABET)
        l_seq = read_labels(label_file, states=STATES)

        if len(g_seq) != len(l_seq):
            print(f"[WARN] length mismatch for {strain}: genome={len(g_seq)} labels={len(l_seq)} -> skipping")
            continue

        # convert genome chars to ints (default A if unexpected, כמו final_model)
        seq_ints = [nuc_to_idx.get(c, 0) for c in g_seq]

        current_ctx = 0

        for t in range(len(g_seq)):
            s_char = l_seq[t]
            s_idx = state_to_idx.get(s_char, None)
            if s_idx is None:
                continue

            n_idx = seq_ints[t]

            if t == 0:
                start_counts[s_idx] += 1

            emit_counts[s_idx, current_ctx, n_idx] += 1

            if t < len(g_seq) - 1:
                next_s = state_to_idx.get(l_seq[t + 1], None)
                if next_s is not None:
                    trans_counts[s_idx, next_s] += 1

            if order > 0:
                current_ctx = ((current_ctx << 2) | n_idx) & mask

    # normalize
    start_prob = start_counts / start_counts.sum()
    trans_prob = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    emit_prob = emit_counts / emit_counts.sum(axis=2, keepdims=True)

    return start_prob, trans_prob, emit_prob


# ======================
# Predict using final_model.fast_high_order_viterbi
# ======================
def predict_high_order(seq_str, start_prob, trans_prob, emit_prob_3d, order):
    """
    מחזיר רצף מצבים באורך len(seq_str) כ-list של תווים מתוך STATES.
    """
    # seq_str אמור להיות רק A/C/G/T כי read_fasta עם alphabet=ALPHABET מסנן
    seq_ints = np.array([nuc_to_idx.get(c, 0) for c in seq_str], dtype=np.int32)

    eps = 1e-10
    start_log = np.log(start_prob + eps)
    trans_log = np.log(trans_prob + eps)
    emit_log = np.log(emit_prob_3d + eps)

    pred_idx = final_model.fast_high_order_viterbi(seq_ints, start_log, trans_log, emit_log, order)
    pred_states = [STATES[i] for i in pred_idx.tolist()]
    return pred_states


# ======================
# Runner
# ======================
def run_final_model_experiments(order=FINAL_ORDER):
    all_results = []

    for exp_idx, exp in enumerate(EXPERIMENTS_CONFIG, start=1):
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {exp_idx}")
        print(f"{'='*70}")

        for train_group, train_strains in exp["train"].items():
            print(f"\n>>> Training group: {train_group} (order={order})")
            print(f"Strains: {len(train_strains)}")

            # -------- TRAIN --------
            start_prob, trans_prob, emit_prob_3d = train_high_order_from_strains(train_strains, order)

            group_results = []
            gene_length_data = []
            non_coding_data = []

            # -------- TEST --------
            for test_group, test_strains in exp["test"].items():
                experiment_name = f"Train:{train_group} | Test:{test_group}"

                for strain in test_strains:
                    print(f"Testing on {strain}")

                    genome_file = f"{BASE_PATH}/{strain}_genome.fasta"
                    label_file = f"{BASE_PATH}/{strain}_labels.fasta"

                    t_seq = read_fasta(genome_file, alphabet=ALPHABET)
                    t_true_6 = read_labels(label_file, states=STATES)

                    t_pred_6 = predict_high_order(t_seq, start_prob, trans_prob, emit_prob_3d, order)

                    # ---- convert 6-state -> binary C/N for evaluation.py ----
                    t_true = to_binary_cn(t_true_6)
                    t_pred = to_binary_cn(t_pred_6)

                    # ---- Length-based stats (מבוסס על C/N) ----
                    for item in ev.collect_gene_length_stats(t_true, t_pred):
                        item["Experiment"] = experiment_name
                        gene_length_data.append(item)

                    for item in ev.collect_non_coding_length_stats(t_true, t_pred):
                        item["Experiment"] = experiment_name
                        non_coding_data.append(item)

                    # ---- Metrics ----
                    metrics = ev.get_metrics_dict(t_true, t_pred, genome_name=strain)
                    metrics.update({
                        "Train": train_group,
                        "Test": test_group,
                        "Experiment": experiment_name,
                        "Model": MODEL_NAME,
                        "Order": order
                    })

                    group_results.append(metrics)
                    all_results.append(metrics)

                    print(f"  F1 = {metrics['F1_Score']:.4f} | Precision={metrics['Precision']:.4f} | Recall={metrics['Recall']:.4f}")

            # -------- PLOTS --------
            if group_results:
                df = pd.DataFrame(group_results)
                print("\nSummary:")
                print(df[["Train", "Test", "Precision", "Recall", "F1_Score", "IoU"]])

                save_path = PLOTS_BASE / f"order_{order}" / train_group
                save_path.mkdir(parents=True, exist_ok=True)

                ev.plot_grouped_benchmark(group_results, metric="F1_Score", save_path=str(save_path))
                ev.plot_confusion_matrix_grid(group_results, save_path=str(save_path))

                if gene_length_data:
                    ev.plot_success_by_length_binned(gene_length_data, save_path=str(save_path))

                if non_coding_data:
                    ev.plot_non_coding_success_by_length(non_coding_data, save_path=str(save_path))
            else:
                print("[WARN] No results generated for this training group.")

    print("\n✔ All experiments completed")
    return pd.DataFrame(all_results)


if __name__ == "__main__":
    df_all = run_final_model_experiments(order=FINAL_ORDER)

    out_csv = PLOTS_BASE / f"final_model_all_results_order_{FINAL_ORDER}.csv"
    df_all.to_csv(out_csv, index=False)

    print(f"\nSaved full results to: {out_csv}")
