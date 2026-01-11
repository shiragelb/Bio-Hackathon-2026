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
                if i < genome_length:
                    labels[i] = "C"

    return "".join(labels)


# ======================
# Process directories recursively
# ======================

# עוברים על כל התיקיות בתוך data (למשל E.coli, Salmonella...)
for species_dir in DATA_DIR.iterdir():
    if not species_dir.is_dir():
        continue
    
    species_name = species_dir.name
    print(f"\n📂 Entering folder: {species_name}")

    # יצירת תיקייה מקבילה ב-processed_data
    species_out_dir = PROCESSED_DIR / species_name
    species_out_dir.mkdir(parents=True, exist_ok=True)

    # חיפוש קבצי ZIP בתוך התיקייה הספציפית
    zip_files = list(species_dir.glob("*.zip"))
    print(f"   Found {len(zip_files)} zip files for {species_name}")

    for zip_path in zip_files:
        zip_name = zip_path.stem  # למשל "042"
        print(f"   ➜ Processing {zip_name}...")

        # חילוץ בתוך התיקייה של המין הספציפי
        extract_dir = species_dir / zip_name

        # Unzip if needed
        if not extract_dir.exists():
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)

        assemblies_root = extract_dir / "ncbi_dataset" / "data"
        if not assemblies_root.exists():
            print("      No ncbi_dataset found, skipping")
            continue

        gcf_dirs = [d for d in assemblies_root.iterdir()
                    if d.is_dir() and d.name.startswith("GCF_")]

        for gcf_dir in gcf_dirs:
            fna_files = list(gcf_dir.glob("*_genomic.fna"))
            gff_files = list(gcf_dir.glob("genomic.gff"))

            if not fna_files or not gff_files:
                continue

            fna_path = fna_files[0]
            gff_path = gff_files[0]

            # קריאת הנתונים
            header, genome_seq = read_fasta(fna_path)
            labels_seq = build_labels(gff_path, len(genome_seq))

            # --- יצירת השם החדש ---
            # משתמשים בשם התיקייה (למשל Salmonella) + שם הזיפ
            final_name = f"{species_name}_{zip_name}"
            
            # שמירה בתיקייה הייעודית בתוך processed_data
            out_genome = species_out_dir / f"{final_name}_genome.fasta"
            out_labels = species_out_dir / f"{final_name}_labels.fasta"

            with open(out_genome, "w") as f:
                f.write(header + "\n")
                f.write(genome_seq + "\n")

            with open(out_labels, "w") as f:
                f.write(f">{final_name}_labels\n")
                f.write(labels_seq + "\n")

            print(f"      ✔ Saved: {final_name}")

print("\nAll done.")