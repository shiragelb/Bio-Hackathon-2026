from pathlib import Path
import zipfile

# ======================
# Config
# ======================
DATA_DIR = Path("data\E.coli")
PROCESSED_DIR = Path("processed_data_2")
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
# GFF → Detailed Labels
# ======================
def build_detailed_labels(gff_path, genome_length):
    labels = ["N"] * genome_length
    count_genes = 0

    with open(gff_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue

            if fields[2] != "CDS":
                continue

            if fields[6] != "+":   # רק strand חיובי (כמו בקוד שלך)
                continue

            start = int(fields[3]) - 1
            end = int(fields[4])

            if end > genome_length:
                continue

            if end - start < 6:
                continue

            count_genes += 1

            # Start codon
            labels[start:start+3] = ["S", "S", "S"]

            # Stop codon
            labels[end-3:end] = ["E", "E", "E"]

            # Internal codons: 1,2,3
            state_cycle = ["1", "2", "3"]
            pos = start + 3
            idx = 0

            while pos < end - 3:
                labels[pos] = state_cycle[idx % 3]
                pos += 1
                idx += 1

    print(f"      GFF: labeled {count_genes} genes")
    return "".join(labels)


# ======================
# Main processing loop
# ======================
for species_dir in DATA_DIR.iterdir():
    if not species_dir.is_dir():
        continue

    species_name = species_dir.name
    print(f"\n📂 Processing species: {species_name}")

    species_out_dir = PROCESSED_DIR / species_name
    species_out_dir.mkdir(parents=True, exist_ok=True)

    zip_files = list(species_dir.glob("*.zip"))
    print(f"   Found {len(zip_files)} zip files")

    for zip_path in zip_files:
        zip_name = zip_path.stem
        print(f"   ➜ Processing ZIP: {zip_name}")

        extract_dir = species_dir / zip_name

        # Unzip if needed
        if not extract_dir.exists():
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)

        assemblies_root = extract_dir / "ncbi_dataset" / "data"
        if not assemblies_root.exists():
            print("      No ncbi_dataset/data found, skipping")
            continue

        gcf_dirs = [
            d for d in assemblies_root.iterdir()
            if d.is_dir() and d.name.startswith("GCF_")
        ]

        for gcf_dir in gcf_dirs:
            fna_files = list(gcf_dir.glob("*_genomic.fna"))
            gff_files = list(gcf_dir.glob("genomic.gff"))

            if not fna_files or not gff_files:
                continue

            fna_path = fna_files[0]
            gff_path = gff_files[0]

            header, genome_seq = read_fasta(fna_path)
            labels_seq = build_detailed_labels(gff_path, len(genome_seq))

            final_name = f"{species_name}_{zip_name}"

            out_genome = species_out_dir / f"{final_name}_genome.fasta"
            out_labels = species_out_dir / f"{final_name}_labels.fasta"

            with open(out_genome, "w") as f:
                f.write(header + "\n")
                f.write(genome_seq + "\n")

            with open(out_labels, "w") as f:
                f.write(f">{final_name}_detailed_labels\n")
                f.write(labels_seq + "\n")

            print(f"      ✔ Saved: {final_name}")

print("\n✅ Done. All organisms processed correctly.")
