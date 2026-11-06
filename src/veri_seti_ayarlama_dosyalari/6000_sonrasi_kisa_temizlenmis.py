from pathlib import Path
import csv

INPUT  = Path("../../data/processed/real_final_dataset_cleaned.csv")
FIXED  = Path("../../data/processed/real_final_dataset_cleaned_fixed.csv")
AFTER6 = Path("../../data/processed/real_final_dataset_after6000.csv")

EXPECTED_HEADER = ["hotel_name", "comment_text", "source_url", "source", "rating_auto", "sentiment_label"]

def fix_lines(in_path: Path, out_path: Path):
    fixed_rows = 0
    skipped = 0

    with in_path.open("r", encoding="utf-8", newline="") as f_in, \
         out_path.open("w", encoding="utf-8", newline="") as f_out:

        writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)

        first_line = f_in.readline()
        # Header kontrolü (dosyan header'sız ise bu kısmı kaldırabilirsin)
        if "hotel_name" in first_line and "comment_text" in first_line:
            writer.writerow(EXPECTED_HEADER)
        else:
            # header yoksa ilk satırı veri say
            f_in.seek(0)

        for line_no, line in enumerate(f_in, start=2):
            line = line.strip()
            if not line:
                continue

            # Basit split: tırnaklara güvenmeyip düz metin mantığıyla toparlayacağız
            parts = [p.strip() for p in line.split(",")]

            # Minimum 6 alan yoksa geç
            if len(parts) < 6:
                skipped += 1
                continue

            hotel_name = parts[0]
            sentiment_label = parts[-1]
            rating_auto = parts[-2]
            source = parts[-3]
            source_url = parts[-4]

            comment_text = ",".join(parts[1:-4]).strip()

            # Küçük doğrulamalar (isteğe bağlı)
            if rating_auto not in {"1","2","3","4","5","6","7","8","9","10"}:
                # rating bozuksa yine de yazmak istersen bu if'i kaldır
                pass

            writer.writerow([hotel_name, comment_text, source_url, source, rating_auto, sentiment_label])
            fixed_rows += 1

    print(f"Fixed rows: {fixed_rows}, skipped: {skipped}")
    print(f"Fixed file: {out_path}")

def split_after6000(in_path: Path, out_path: Path, start_idx: int = 6000):
    with in_path.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.reader(f_in)
        rows = list(reader)

    header = rows[0]
    data = rows[1:]

    data_after = data[start_idx:]

    with out_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        writer.writerows(data_after)

    print(f"After6000 rows saved: {len(data_after)}")
    print(f"After6000 file: {out_path}")

if __name__ == "__main__":
    fix_lines(INPUT, FIXED)
    split_after6000(FIXED, AFTER6, start_idx=6000)