from hmm import (
    read_fasta, read_labels,
    train_supervised_hmm, viterbi, metrics
)

# ======================
# Paths
# ======================
BASE = "processed_data"

# הפורמט החדש: Escherichia_coli_{Original_Zip_Name}_{Type}.fasta
# ודאי ששמות הקבצים כאן תואמים בדיוק לשמות קבצי ה-ZIP שהיו לך (בלי הסיומת .zip)

TRAIN = [
    # Format: (Genome File, Label File)
    ("Escherichia_coli_K12-MG1655_genome.fasta", "Escherichia_coli_K12-MG1655_labels.fasta"), # K-12
    ("Escherichia_coli_E. coli B REL606_genome.fasta",      "Escherichia_coli_E. coli B REL606_labels.fasta"),      # REL606
    ("Escherichia_coli_HS_genome.fasta",          "Escherichia_coli_HS_labels.fasta"),          # HS
    ("Escherichia_coli_SE11_genome.fasta",        "Escherichia_coli_SE11_labels.fasta"),        # SE11
]

TEST = (
    "Escherichia_coli_042_genome.fasta",  # 042 (Test Strain)
    "Escherichia_coli_042_labels.fasta"
)

# ======================
# Build training data
# ======================
train_seq = ""
train_labels = ""

print("Building training set...")
try:
    for genome, labels in TRAIN:
        print(f"  Loading {genome}...")
        s = read_fasta(f"{BASE}/{genome}")
        y = read_labels(f"{BASE}/{labels}")
        
        if len(s) != len(y):
            raise ValueError(f"Length mismatch in {genome}: seq={len(s)}, labels={len(y)}")

        train_seq += s
        train_labels += y

    print(f"Total training length: {len(train_seq):,}")

    # ======================
    # Train HMM
    # ======================
    print("\nTraining HMM...")
    emissions, transitions, init = train_supervised_hmm(
        train_seq, train_labels, laplace=1.0
    )

    print("Training completed.")
    print("Transitions:", transitions)
    print("Emissions:", emissions)

    # ======================
    # Test
    # ======================
    print(f"\nEvaluating on test strain: {TEST[0]}...")

    test_seq = read_fasta(f"{BASE}/{TEST[0]}")
    test_labels = read_labels(f"{BASE}/{TEST[1]}")

    pred = viterbi(test_seq, emissions, transitions, init)
    res = metrics(test_labels, pred)

    print("\nTest results:")
    for k, v in res.items():
        print(f"{k:10s}: {v:.4f}")

except FileNotFoundError as e:
    print(f"\nError: Could not find file. Please check the filenames in 'processed_data'.\n{e}")
except Exception as e:
    print(f"\nAn error occurred:\n{e}")