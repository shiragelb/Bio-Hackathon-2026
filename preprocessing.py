from pathlib import Path
import zipfile

DATA_DIR = Path("data")
PROCESSED_DIR = Path("processed_data")
PROCESSED_DIR.mkdir(exist_ok=True)

# ======================
# FASTA reader
# ======================
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


# ======================
# GFF → label sequence
# ======================
def build_labels(gff_path, genome_length):
    labels = ["N"] * genome_length

    with open(gff_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue

            if fields[2] != "CDS":
                continue

            start = int(fields[3]) - 1
            end = int(fields[4])

            for i in range(start, end):
                labels[i] = "C"

    return "".join(labels)


# ======================
# Process all ZIPs
# ======================
zip_files = list(DATA_DIR.glob("*.zip"))
print(f"Found {len(zip_files)} zip files")

for zip_path in zip_files:
    print(f"\n=== Processing {zip_path.name} ===")

    extract_dir = DATA_DIR / zip_path.stem

    # unzip if needed
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    assemblies_root = extract_dir / "ncbi_dataset" / "data"
    if not assemblies_root.exists():
        print("  No ncbi_dataset found, skipping")
        continue

    gcf_dirs = [d for d in assemblies_root.iterdir()
                if d.is_dir() and d.name.startswith("GCF_")]

    print(f"  Found {len(gcf_dirs)} GCF assemblies")

    for gcf_dir in gcf_dirs:
        fna_files = list(gcf_dir.glob("*_genomic.fna"))
        gff_files = list(gcf_dir.glob("genomic.gff"))

        if not fna_files or not gff_files:
            print(f"  Skipping {gcf_dir.name} (missing files)")
            continue

        fna_path = fna_files[0]
        gff_path = gff_files[0]

        header, genome_seq = read_fasta(fna_path)
        labels_seq = build_labels(gff_path, len(genome_seq))

        out_genome = PROCESSED_DIR / f"{gcf_dir.name}_genome.fasta"
        out_labels = PROCESSED_DIR / f"{gcf_dir.name}_labels.fasta"

        with open(out_genome, "w") as f:
            f.write(header + "\n")
            f.write(genome_seq + "\n")

        with open(out_labels, "w") as f:
            f.write(f">{gcf_dir.name}_labels\n")
            f.write(labels_seq + "\n")

        print(f"  ✔ {gcf_dir.name} | length={len(genome_seq)}")

print("\nAll done.")
