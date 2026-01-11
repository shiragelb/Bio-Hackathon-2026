from pathlib import Path
import zipfile
import re

# ======================
# Config
# ======================
DATA_DIR = Path("data")
PROCESSED_DIR = Path("processed_data_2")
PROCESSED_DIR.mkdir(exist_ok=True)

# ======================
# Helper: Clean Name
# ======================
def get_safe_name_from_header(header):
    if not header:
        return "unknown_organism"
    parts = header.lstrip(">").split(" ", 1)
    if len(parts) < 2:
        return parts[0]
    raw_name = parts[1].split(",")[0]
    safe_name = re.sub(r'[^\w\- ]', '', raw_name) 
    safe_name = safe_name.replace(" ", "_")
    return safe_name

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
# GFF → Detailed Labels (Safe Version with Sanity Checks)
# ======================
def build_detailed_labels(gff_path, genome_seq):
    # חישוב אורך הגנום מתוך הרצף שהתקבל
    genome_length = len(genome_seq)
    
    # אתחול הכל כ-Non-coding
    labels = ["N"] * genome_length

    # סטים של קודונים חוקיים
    VALID_STARTS = {"ATG", "GTG", "TTG"} 
    VALID_STOPS = {"TAA", "TAG", "TGA"}

    # מונים לסטטיסטיקה
    stats = {"ok": 0, "bad_start": 0, "bad_stop": 0, "partial": 0, "skipped_strand": 0}

    with open(gff_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue

            # סינון: רק אזורים מקודדים (CDS)
            if fields[2] != "CDS":
                continue
            
            # סינון: רק גדיל חיובי (+)
            if fields[6] != "+":
                stats["skipped_strand"] += 1
                continue

            # המרה ל-0-based
            start = int(fields[3]) - 1
            end = int(fields[4])
            
            # בדיקת גבולות בסיסית
            if end > genome_length:
                continue
            
            # בדיקת אורך (חייב להתחלק ב-3)
            if (end - start) % 3 != 0:
                continue

            # === Sanity Check: בדיקה מול הרצף האמיתי ===
            gene_seq = genome_seq[start:end]
            actual_start = gene_seq[:3]
            actual_stop = gene_seq[-3:]

            # בדיקה אם זה גן חלקי (לפי המטא-דאטה)
            attributes = fields[8]
            if "partial=true" in attributes:
                stats["partial"] += 1
                # כאן מחליטים אם לכלול או לא. ליתר ביטחון נדלג אם זה לא נראה תקין:
                if actual_start not in VALID_STARTS or actual_stop not in VALID_STOPS:
                     continue

            # בדיקת קודונים
            if actual_start not in VALID_STARTS:
                stats["bad_start"] += 1
                continue # מדלגים על דאטה שגוי!

            if actual_stop not in VALID_STOPS:
                stats["bad_stop"] += 1
                continue # מדלגים על דאטה שגוי!

            stats["ok"] += 1

            # === תיוג המבנה הפנימי ===
            
            # 1. Start Codon -> S
            labels[start] = "S"
            labels[start+1] = "S"
            labels[start+2] = "S"

            # 2. Stop Codon -> E
            labels[end-3] = "E"
            labels[end-2] = "E"
            labels[end-1] = "E"

            # 3. Internal Codons (1, 2, 3)
            current_pos = start + 3
            stop_pos = end - 3
            
            while current_pos < stop_pos:
                labels[current_pos] = "1"
                labels[current_pos+1] = "2"
                labels[current_pos+2] = "3"
                current_pos += 3

    # הדפסת הסטטיסטיקה עבור הקובץ הנוכחי
    print(f"    Sanity Check: Valid Genes={stats['ok']}, Bad Start={stats['bad_start']}, Bad Stop={stats['bad_stop']}")
    
    return "".join(labels)

# ======================
# Process all ZIPs
# ======================
zip_files = list(DATA_DIR.glob("*.zip"))
print(f"Found {len(zip_files)} zip files in {DATA_DIR}")
print(f"Output directory: {PROCESSED_DIR}")

if len(zip_files) == 0:
    print("Warning: No zip files found! Check if the 'data' folder exists and contains .zip files.")

for zip_path in zip_files:
    zip_name = zip_path.stem
    print(f"\n=== Processing {zip_name} ===")

    extract_dir = DATA_DIR / zip_name

    # Unzip if needed
    if not extract_dir.exists():
        print(f"  Extracting {zip_name}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    assemblies_root = extract_dir / "ncbi_dataset" / "data"
    if not assemblies_root.exists():
        print("  No ncbi_dataset found inside zip, skipping")
        continue

    gcf_dirs = [d for d in assemblies_root.iterdir()
                if d.is_dir() and d.name.startswith("GCF_")]

    for gcf_dir in gcf_dirs:
        fna_files = list(gcf_dir.glob("*_genomic.fna"))
        gff_files = list(gcf_dir.glob("genomic.gff"))

        if not fna_files or not gff_files:
            print(f"  Missing fna or gff in {gcf_dir.name}")
            continue

        fna_path = fna_files[0]
        gff_path = gff_files[0]

        # קריאה
        header, genome_seq = read_fasta(fna_path)
        
        # === כאן השינוי בקריאה לפונקציה ===
        # אנחנו מעבירים את genome_seq עצמו ולא רק את האורך
        labels_seq = build_detailed_labels(gff_path, genome_seq)

        # שמות קבצים
        final_name = f"Escherichia_coli_{zip_name}"
        out_genome = PROCESSED_DIR / f"{final_name}_genome.fasta"
        out_labels = PROCESSED_DIR / f"{final_name}_labels.fasta"

        # שמירה
        with open(out_genome, "w") as f:
            f.write(header + "\n")
            f.write(genome_seq + "\n")

        with open(out_labels, "w") as f:
            f.write(f">{final_name}_detailed_labels_S_E_123\n")
            f.write(labels_seq + "\n")

        print(f"  ✔ Saved: {final_name}")

print("\nDone. Files ready in processed_data_2/")