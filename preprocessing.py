from pathlib import Path

# ---- paths (for now hard-coded) ----
FASTA_PATH = Path(r"ncbi_dataset\ncbi_dataset\data\GCF_000005845.2\GCF_000005845.2_ASM584v2_genomic.fna")
GFF_PATH   = Path(r"ncbi_dataset\ncbi_dataset\data\GCF_000005845.2\genomic.gff")

OUT_FASTA = Path("ecoli_genome.fasta")
OUT_LABELS = Path("ecoli_labels.fasta")


# ---- read FASTA ----
def read_fasta(path):
    header = None
    seq = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                header = line
            else:
                seq.append(line.upper())

    return header, "".join(seq)


# ---- parse GFF and build labels ----
def build_labels(gff_path, genome_length):
    labels = ["N"] * genome_length

    with open(gff_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue

            feature_type = fields[2]
            if feature_type != "CDS":
                continue

            start = int(fields[3]) - 1  # to 0-based
            end = int(fields[4])        # inclusive → exclusive

            for i in range(start, end):
                labels[i] = "C"

    return "".join(labels)


# ---- main ----
header, genome_seq = read_fasta(FASTA_PATH)
labels_seq = build_labels(GFF_PATH, len(genome_seq))

# write genome
with open(OUT_FASTA, "w") as f:
    f.write(header + "\n")
    f.write(genome_seq + "\n")

# write labels
with open(OUT_LABELS, "w") as f:
    f.write(">NC_000913.3_labels\n")
    f.write(labels_seq + "\n")

print("Done.")
print(f"Genome length: {len(genome_seq)}")
print(f"Labels length: {len(labels_seq)}")
print(f"Coding fraction: {labels_seq.count('C') / len(labels_seq):.3f}")