from pathlib import Path
import json
import random
import pandas as pd


"""
Build JSONL files from grouped pose CSVs.

Input:
- X_grouped.csv
- y_grouped.csv

Output:
- train.jsonl
- val.jsonl
- test.jsonl
"""


LABEL_TEXT = {
    0: "0",
    1: "1",
}


def row_to_prompt(row: pd.Series) -> str:
    """
    Convert one grouped row into a prompt string.
    """
    values = [f"{float(v):.6f}" for v in row.tolist()]
    pose_text = ",".join(values)

    return (
        "You are a binary stability classifier.\n"
        "Input is one 30-frame pose window.\n"
        "Return only one token: 0 or 1.\n"
        "0 means stable.\n"
        "1 means unstable.\n"
        f"Pose window:\n{pose_text}"
    )


def build_records(x_df: pd.DataFrame, y_df: pd.Series) -> list[dict]:
    """
    Convert CSV rows into chat-format records.
    """
    records = []

    for idx in range(len(x_df)):
        prompt = row_to_prompt(x_df.iloc[idx])
        label = int(y_df.iloc[idx])

        record = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": LABEL_TEXT[label]},
            ]
        }
        records.append(record)

    return records


def split_records(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split records into train, val, and test.
    """
    random.seed(seed)
    shuffled = records[:]
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    return train, val, test


def write_jsonl(path: Path, records: list[dict]) -> None:
    """
    Write one JSON object per line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """
    Main entry point.
    """
    data_dir = Path("data")
    x_path = data_dir / "X_grouped.csv"
    y_path = data_dir / "y_grouped.csv"

    x_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path).iloc[:, 0]

    if len(x_df) != len(y_df):
        raise ValueError("X and y row counts do not match.")

    records = build_records(x_df, y_df)
    train, val, test = split_records(
        records=records,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
    )

    write_jsonl(data_dir / "train.jsonl", train)
    write_jsonl(data_dir / "val.jsonl", val)
    write_jsonl(data_dir / "test.jsonl", test)

    print(f"Total records: {len(records)}")
    print(f"Train records: {len(train)}")
    print(f"Val records: {len(val)}")
    print(f"Test records: {len(test)}")


if __name__ == "__main__":
    main()