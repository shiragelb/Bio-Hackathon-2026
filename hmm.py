import math
from collections import defaultdict

def read_fasta(path: str) -> str:
    seq_parts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq_parts.append(line.upper())
    seq = "".join(seq_parts)

    # תיקון: במקום לזרוק שגיאה על אותיות לא מוכרות, נמיר אותן ל-N
    # W, R, Y, M, K, S, etc. -> N
    valid_chars = set("ACGT")
    cleaned_seq = []
    for ch in seq:
        if ch in valid_chars:
            cleaned_seq.append(ch)
        else:
            cleaned_seq.append("N")
            
    return "".join(cleaned_seq)

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

def train_supervised_hmm(seq: str, labels: str, states, alphabet, laplace: float = 1.0):
    seq = list(seq)
    labels = list(labels)

    if len(seq) != len(labels):
        raise ValueError(f"len(seq)={len(seq)} != len(labels)={len(labels)}")

    emit_counts = {s: defaultdict(int) for s in states}
    trans_counts = {s: defaultdict(int) for s in states}
    init_counts = defaultdict(int)

    # ספירת מעברים ופליטות
    init_counts[labels[0]] += 1
    for t, (x, y) in enumerate(zip(seq, labels)):
        emit_counts[y][x] += 1
        if t > 0:
            trans_counts[labels[t - 1]][y] += 1

    emissions = {s: {} for s in states}
    transitions = {s: {} for s in states}

    # חישוב הסתברויות עם החלקה (Laplace Smoothing)
    for s in states:
        # סך הכל תווים שנפלטו ממצב s + החלקה לכל תו באלפבית (כולל N)
        total = sum(emit_counts[s][a] for a in alphabet) + laplace * len(alphabet)
        for a in alphabet:
            # אם התו a לא הופיע מעולם, הוא יקבל הסתברות קטנה בזכות ה-laplace
            count = emit_counts[s][a]
            emissions[s][a] = (count + laplace) / total

    for sp in states:
        total = sum(trans_counts[sp][s] for s in states) + laplace * len(states)
        for s in states:
            transitions[sp][s] = (trans_counts[sp][s] + laplace) / total

    # חישוב הסתברויות התחלה
    init = {s: laplace for s in states}
    init[labels[0]] += 1.0
    z = sum(init.values())
    init = {s: p / z for s, p in init.items()}

    return emissions, transitions, init

def viterbi(seq: str, emissions, transitions, init, states):
    T = len(seq)
    log = math.log

    dp = {s: [-math.inf] * T for s in states}
    back = {s: [None] * T for s in states}

    # אתחול (t=0)
    x0 = seq[0]
    for s in states:
        # אם x0 הוא תו שלא ראינו באימון (למשל N שלא הופיע), ה-emissions יטפל בזה
        # כי הוספנו את N לאלפבית והשתמשנו ב-Laplace smoothing
        dp[s][0] = log(init[s]) + log(emissions[s][x0])

    # רקורסיה
    for t in range(1, T):
        xt = seq[t]
        for s in states:
            best_sp, best_score = None, -math.inf
            for sp in states:
                score = dp[sp][t - 1] + log(transitions[sp][s]) + log(emissions[s][xt])
                if score > best_score:
                    best_score = score
                    best_sp = sp
            dp[s][t] = best_score
            back[s][t] = best_sp

    # סיום וחזרה לאחור (Backtracking)
    last = max(states, key=lambda s: dp[s][T - 1])
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
    # בדיקה מקומית אם מריצים את הקובץ ישירות
    # (לא ירוץ כשאת מייבאת אותו ב-experiment1)
    pass
