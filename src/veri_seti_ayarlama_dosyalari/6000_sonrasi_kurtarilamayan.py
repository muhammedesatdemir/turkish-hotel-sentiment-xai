from pathlib import Path
import csv

INPUT_PATH = Path("../../data/processed/real_final_dataset_cleaned.csv")
FIXED_PATH = Path("../../data/processed/real_final_dataset_cleaned_fixed.csv")
BAD_LINES_PATH = Path("../../data/processed/bad_lines_detected.csv")
AFTER6000_PATH = Path("../../data/processed/real_final_dataset_after6000.csv")

HEADER = [
    "hotel_name",
    "comment_text",
    "source_url",
    "source",
    "rating_auto",
    "sentiment_label"
]


def fix_csv():

    fixed_rows = 0
    skipped_rows = 0
    total_rows = 0

    with INPUT_PATH.open("r", encoding="utf-8", newline="") as f_in, \
         FIXED_PATH.open("w", encoding="utf-8", newline="") as f_out, \
         BAD_LINES_PATH.open("w", encoding="utf-8", newline="") as bad_out:

        writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)
        bad_writer = csv.writer(bad_out)

        writer.writerow(HEADER)
        bad_writer.writerow(["line_number", "raw_line"])

        for line_number, line in enumerate(f_in, start=1):

            total_rows += 1
            line = line.strip()

            if not line:
                skipped_rows += 1
                bad_writer.writerow([line_number, "EMPTY_LINE"])
                continue

            parts = [p.strip() for p in line.split(",")]

            if len(parts) < 6:
                skipped_rows += 1
                bad_writer.writerow([line_number, line])
                continue

            try:

                hotel_name = parts[0]
                sentiment_label = parts[-1]
                rating_auto = parts[-2]
                source = parts[-3]
                source_url = parts[-4]

                comment_text = ",".join(parts[1:-4]).strip()

                writer.writerow([
                    hotel_name,
                    comment_text,
                    source_url,
                    source,
                    rating_auto,
                    sentiment_label
                ])

                fixed_rows += 1

            except Exception:
                skipped_rows += 1
                bad_writer.writerow([line_number, line])

    print("\n====== CSV FIX REPORT ======")
    print("Toplam satır:", total_rows)
    print("Düzeltilen satır:", fixed_rows)
    print("Atlanan satır:", skipped_rows)
    print("Fixed CSV:", FIXED_PATH)
    print("Bad lines file:", BAD_LINES_PATH)


def split_after6000():

    with FIXED_PATH.open("r", encoding="utf-8", newline="") as f:

        reader = csv.reader(f)
        rows = list(reader)

    header = rows[0]
    data = rows[1:]

    after6000 = data[6000:]

    with AFTER6000_PATH.open("w", encoding="utf-8", newline="") as f_out:

        writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        writer.writerows(after6000)

    print("\n====== SPLIT REPORT ======")
    print("6000 sonrası satır:", len(after6000))
    print("Yeni dosya:", AFTER6000_PATH)


if __name__ == "__main__":

    fix_csv()
    split_after6000()