import math
from collections import defaultdict

STATES = ("C", "N")
ALPHABET = ("A", "C", "G", "T")

def read_fasta(path: str) -> str:
    seq_parts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq_parts.append(line.upper())
    seq = "".join(seq_parts)

    bad = {ch for ch in seq if ch not in set(ALPHABET)}
    if bad:
        raise ValueError(f"FASTA contains non-ACGT characters: {sorted(bad)}")
    return seq

def read_labels(path: str) -> str:
    parts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().upper()
            if not line or line.startswith(">"):
                continue
            parts.append("".join(line.split()))
    s = "".join(parts)

    bad = {ch for ch in s if ch not in {"C", "N"}}
    if bad:
        raise ValueError(f"Labels contain invalid characters: {sorted(bad)} (allowed: C/N)")
    return s

def train_supervised_hmm(seq: str, labels: str, laplace: float = 1.0):
    seq = list(seq)
    labels = list(labels)

    if len(seq) != len(labels):
        raise ValueError(f"len(seq)={len(seq)} != len(labels)={len(labels)}")

    emit_counts = {s: defaultdict(int) for s in STATES}
    trans_counts = {s: defaultdict(int) for s in STATES}
    init_counts = defaultdict(int)

    init_counts[labels[0]] += 1
    for t, (x, y) in enumerate(zip(seq, labels)):
        emit_counts[y][x] += 1
        if t > 0:
            trans_counts[labels[t - 1]][y] += 1

    emissions = {s: {} for s in STATES}
    transitions = {s: {} for s in STATES}

    for s in STATES:
        total = sum(emit_counts[s][a] for a in ALPHABET) + laplace * len(ALPHABET)
        for a in ALPHABET:
            emissions[s][a] = (emit_counts[s][a] + laplace) / total

    for sp in STATES:
        total = sum(trans_counts[sp][s] for s in STATES) + laplace * len(STATES)
        for s in STATES:
            transitions[sp][s] = (trans_counts[sp][s] + laplace) / total

    init = {s: laplace for s in STATES}
    init[labels[0]] += 1.0
    z = sum(init.values())
    init = {s: p / z for s, p in init.items()}

    return emissions, transitions, init

def viterbi(seq: str, emissions, transitions, init):
    T = len(seq)
    log = math.log

    dp = {s: [-math.inf] * T for s in STATES}
    back = {s: [None] * T for s in STATES}

    x0 = seq[0]
    for s in STATES:
        dp[s][0] = log(init[s]) + log(emissions[s][x0])

    for t in range(1, T):
        xt = seq[t]
        for s in STATES:
            best_sp, best_score = None, -math.inf
            for sp in STATES:
                score = dp[sp][t - 1] + log(transitions[sp][s]) + log(emissions[s][xt])
                if score > best_score:
                    best_score = score
                    best_sp = sp
            dp[s][t] = best_score
            back[s][t] = best_sp

    last = max(STATES, key=lambda s: dp[s][T - 1])
    path = [last]
    for t in range(T - 1, 0, -1):
        last = back[last][t]
        path.append(last)
    path.reverse()
    return path

def metrics(y_true, y_pred, positive="C"):
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yp == positive and yt == positive: tp += 1
        elif yp == positive and yt != positive: fp += 1
        elif yp != positive and yt != positive: tn += 1
        else: fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    acc  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

if __name__ == "__main__":
    fasta_path = "genome.fna"
    labels_path = "labels_CN.txt"

    seq = read_fasta(fasta_path)
    labels = read_labels(labels_path)

    emissions, transitions, init = train_supervised_hmm(seq, labels, laplace=1.0)
    pred = viterbi(seq, emissions, transitions, init)

    print("Transitions:", transitions)
    print("Emissions:", emissions)
    print("Metrics:", metrics(labels, pred))
