import csv
from pathlib import Path

INPUT = Path(r"..\..\data\processed\recovered_bad_lines.csv")
OUTPUT = Path(r"..\..\data\processed\recovered_bad_lines_clean.csv")

EXPECTED_COLS = 6

def parse_relaxed_csv(text: str):
    """
    CSV parser that is tolerant to stray quotes inside quoted fields.
    Rules:
      - Comma ends a field only when not in quotes.
      - Newline ends a record only when not in quotes.
      - Quote handling:
          * If inside quotes and next char is quote => escaped quote.
          * If inside quotes and next char is comma/newline/EOF => closing quote.
          * Otherwise => treat as a literal quote (stray quote).
          * If not inside quotes:
              - quote starts quoted field only if field is empty; otherwise literal.
    """
    rows = []
    fields = []
    buf = []

    in_quotes = False

    i = 0
    n = len(text)

    def end_field():
        nonlocal buf, fields
        fields.append("".join(buf))
        buf = []

    def end_row():
        nonlocal fields
        # ignore completely empty last line
        if fields or buf:
            end_field()
            rows.append(fields)
        fields = []

    while i < n:
        ch = text[i]

        if ch == '"':
            nxt = text[i + 1] if i + 1 < n else None

            if in_quotes:
                # Escaped quote
                if nxt == '"':
                    buf.append('"')
                    i += 2
                    continue

                # Valid closing quote only if followed by delimiter / newline / EOF
                if nxt in (",", "\n", "\r", None):
                    in_quotes = False
                    i += 1
                    continue

                # Otherwise stray quote inside quoted field -> keep it
                buf.append('"')
                i += 1
                continue

            else:
                # Start quoted field only if field buffer is empty
                if len(buf) == 0:
                    in_quotes = True
                    i += 1
                    continue
                # stray quote in unquoted field -> keep
                buf.append('"')
                i += 1
                continue

        if ch == "," and not in_quotes:
            end_field()
            i += 1
            continue

        # Handle CRLF / LF
        if ch == "\n" and not in_quotes:
            end_row()
            i += 1
            continue

        if ch == "\r":
            # swallow \r, let \n handle row ending
            i += 1
            continue

        # normal char
        buf.append(ch)
        i += 1

    # flush last row
    if buf or fields:
        end_row()

    return rows


def normalize_row(row):
    # Trim BOM or weird spaces
    return [c.strip("\ufeff") for c in row]

def main():
    raw = INPUT.read_text(encoding="utf-8", errors="replace")
    rows = parse_relaxed_csv(raw)

    # Header check
    header = normalize_row(rows[0])
    data_rows = rows[1:]

    # Fix rows with extra columns by merging middle parts into comment_text (rare)
    cleaned = []
    bad = 0
    for r in data_rows:
        r = normalize_row(r)

        if len(r) == EXPECTED_COLS:
            cleaned.append(r)
            continue

        # If too many columns, try to merge columns 1..k back into comment_text
        if len(r) > EXPECTED_COLS:
            bad += 1
            hotel = r[0]
            source_url = r[-4]
            source = r[-3]
            rating = r[-2]
            label = r[-1]
            comment = ",".join(r[1:-4])  # merge back
            cleaned.append([hotel, comment, source_url, source, rating, label])
            continue

        # If too few columns, pad (also rare)
        if len(r) < EXPECTED_COLS:
            bad += 1
            r = r + [""] * (EXPECTED_COLS - len(r))
            cleaned.append(r)

    # OPTIONAL: clean the specific artifact pattern inside comment_text:
    # remove standalone doubled quotes around line breaks that came from the source
    for r in cleaned:
        txt = r[1]
        txt = txt.replace('\n""', '\n').replace('""\n', '\n')
        # also collapse triple quotes artifacts if any
        txt = txt.replace('"""', '"')
        r[1] = txt

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(header)
        w.writerows(cleaned)

    print(f"Input rows (incl header): {len(rows)}")
    print(f"Output rows (excl header): {len(cleaned)}")
    print(f"Rows needing repair (len != {EXPECTED_COLS}): {bad}")
    print(f"Saved: {OUTPUT}")

if __name__ == "__main__":
    main()