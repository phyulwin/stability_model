from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_TEXT = {0: "0", 1: "1"}


def row_to_prompt(row: pd.Series) -> str:
    values = [f"{float(v):.3f}" for v in row.tolist()]
    pose_text = ",".join(values)
    return (
        "You are a binary stability classifier.\n"
        "Return only one token: 0 or 1.\n"
        "0 = stable\n"
        "1 = unstable\n"
        f"Pose window:\n{pose_text}"
    )


def build_records(x_df: pd.DataFrame, y_series: pd.Series) -> list[dict]:
    records = []
    for i in range(len(x_df)):
        label = int(y_series.iloc[i])
        if label not in LABEL_TEXT:
            raise ValueError(f"Unexpected label at row {i}: {label}")
        records.append(
            {
                "messages": [
                    {"role": "user", "content": row_to_prompt(x_df.iloc[i])},
                    {"role": "assistant", "content": LABEL_TEXT[label]},
                ]
            }
        )
    return records


def extract_labels(records: list[dict]) -> list[int]:
    return [int(r["messages"][1]["content"]) for r in records]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    data_dir = Path("data")
    x_path = data_dir / "X_grouped.csv"
    y_path = data_dir / "y_grouped.csv"

    x_df = pd.read_csv(x_path)
    y_series = pd.read_csv(y_path).iloc[:, 0]

    if len(x_df) != len(y_series):
        raise ValueError("X_grouped.csv and y_grouped.csv row counts do not match.")

    records = build_records(x_df, y_series)
    labels = extract_labels(records)

    train_records, temp_records = train_test_split(
        records,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    temp_labels = extract_labels(temp_records)
    val_records, test_records = train_test_split(
        temp_records,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels,
    )

    write_jsonl(data_dir / "train.jsonl", train_records)
    write_jsonl(data_dir / "val.jsonl", val_records)
    write_jsonl(data_dir / "test.jsonl", test_records)

    print(f"Total: {len(records)}")
    print(f"Train: {len(train_records)}")
    print(f"Val:   {len(val_records)}")
    print(f"Test:  {len(test_records)}")


if __name__ == "__main__":
    main()
