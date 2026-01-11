from pathlib import Path
import zipfile
import re

# ======================
# Config
# ======================
DATA_DIR = Path("data\E.coli")
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
# GFF → Detailed Labels (Noisy / Trust-GFF Version)
# ======================
def build_detailed_labels_noisy(gff_path, genome_length):
    """
    מייצרת לייבלים אך ורק לפי הקואורדינטות ב-GFF.
    ללא סינון של קודוני התחלה/סוף וללא בדיקת חלוקה ב-3.
    """
    # אתחול הכל כ-Non-coding
    labels = ["N"] * genome_length

    # מונה סטטיסטיקה פשוטה בלבד
    count_genes = 0

    with open(gff_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue

            # סינון בסיסי שחייבים: רק CDS ורק גדיל חיובי (כי המודל שלנו חד-כיווני)
            if fields[2] != "CDS":
                continue
            
            if fields[6] != "+":
                continue

            # המרה ל-0-based
            start = int(fields[3]) - 1
            end = int(fields[4])
            length = end - start
            
            # בדיקת גבולות בסיסית (כדי שהקוד לא יקרוס)
            if end > genome_length:
                continue
            
            # בדיקת אורך מינימלית: חייבים לפחות 6 אותיות (3 להתחלה, 3 לסוף)
            # אם הגן קצר מ-6, אי אפשר פיזית לשים S ו-E בלי שידרסו אחד את השני.
            if length < 6:
                continue

            count_genes += 1

            # === תיוג המבנה הפנימי (בלי שאלות) ===
            
            # 1. Start Codon -> S (תמיד 3 הראשונים)
            labels[start] = "S"
            labels[start+1] = "S"
            labels[start+2] = "S"

            # 2. Stop Codon -> E (תמיד 3 האחרונים)
            labels[end-3] = "E"
            labels[end-2] = "E"
            labels[end-1] = "E"

            # 3. Internal Codons (1, 2, 3)
            # ממלאים את האמצע במחזוריות 1,2,3
            current_pos = start + 3
            stop_pos = end - 3
            
            # אם האורך לא מתחלק ב-3, המחזוריות פשוט תיעצר איפה שהיא תיעצר (למשל ב-1 או ב-2)
            # והמודל ילמד להתמודד עם זה.
            state_cycle = ["1", "2", "3"]
            cycle_idx = 0
            
            while current_pos < stop_pos:
                labels[current_pos] = state_cycle[cycle_idx % 3]
                current_pos += 1
                cycle_idx += 1

    print(f"    GFF Processing: Labeled {count_genes} genes (Trusted GFF blindly).")
    return "".join(labels)

# ======================
# Process all ZIPs
# ======================
zip_files = list(DATA_DIR.glob("*.zip"))
print(f"Found {len(zip_files)} zip files in {DATA_DIR}")
print(f"Output directory: {PROCESSED_DIR}")

for zip_path in zip_files:
    zip_name = zip_path.stem
    print(f"\n=== Processing {zip_name} ===")

    extract_dir = DATA_DIR / zip_name

    # Unzip if needed
    if not extract_dir.exists():
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
            continue

        fna_path = fna_files[0]
        gff_path = gff_files[0]

        # קריאה (צריך את ה-FASTA רק בשביל האורך וה-Header)
        header, genome_seq = read_fasta(fna_path)
        
        # === קריאה לפונקציה ה"רועשת" החדשה ===
        # מעבירים רק את האורך, כי לא מעניין אותנו התוכן (בדיקות תקינות)
        labels_seq = build_detailed_labels_noisy(gff_path, len(genome_seq))

        # שמות קבצים
        final_name = f"Escherichia_coli_{zip_name}"
        out_genome = PROCESSED_DIR / f"{final_name}_genome.fasta"
        out_labels = PROCESSED_DIR / f"{final_name}_labels.fasta"

        # שמירה (דורס את הקבצים הקודמים)
        with open(out_genome, "w") as f:
            f.write(header + "\n")
            f.write(genome_seq + "\n")

        with open(out_labels, "w") as f:
            f.write(f">{final_name}_detailed_labels_S_E_123_NO_FILTER\n")
            f.write(labels_seq + "\n")

        print(f"  ✔ Overwritten/Saved: {final_name}")

print("\nDone. Files updated in processed_data_2/ (No filtering applied)")