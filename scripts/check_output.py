"""
check_output.py
===============
Kiểm tra chất lượng output sau khi chạy pipeline.

Kiểm tra:
  - File tồn tại đúng format
  - sentence_id đúng chuẩn và không trùng
  - NER label hợp lệ
  - Thống kê phân phối nhãn

Cách chạy:
  python scripts/check_output.py --id HVH_001_01
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import print as rprint

VALID_LABELS = {"PER", "LOC", "ORG", "TITLE", "TME", "NUM", "DYNASTY"}

console = Console()


def check_seg_tsv(path: Path) -> dict:
    """Kiểm tra file _seg.tsv."""
    issues = []
    records = []
    seen_ids = set()

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            parts = line.split("\t")

            if len(parts) != 2:
                issues.append(f"Dòng {lineno}: thiếu tab, có {len(parts)} cột")
                continue

            sent_id, sentence = parts
            if sent_id in seen_ids:
                issues.append(f"Dòng {lineno}: sentence_id trùng '{sent_id}'")
            seen_ids.add(sent_id)

            if not sentence.strip():
                issues.append(f"Dòng {lineno}: câu rỗng (id={sent_id})")

            records.append({"id": sent_id, "sentence": sentence})

    return {
        "total":  len(records),
        "issues": issues,
        "records": records
    }


def check_ner_json(path: Path) -> dict:
    """Kiểm tra file _ner.json."""
    issues = []
    label_counts = Counter()
    seen_ids = set()

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    for i, rec in enumerate(data):
        sid = rec.get("sentence_id", f"[record {i}]")

        if sid in seen_ids:
            issues.append(f"sentence_id trùng: '{sid}'")
        seen_ids.add(sid)

        if "sentence" not in rec:
            issues.append(f"{sid}: thiếu trường 'sentence'")
        if "entities" not in rec:
            issues.append(f"{sid}: thiếu trường 'entities'")
            continue

        for ent in rec["entities"]:
            label = ent.get("label", "")
            text  = ent.get("text", "")

            if label not in VALID_LABELS:
                issues.append(f"{sid}: nhãn không hợp lệ '{label}' (text='{text}')")
            else:
                label_counts[label] += 1

            if not text:
                issues.append(f"{sid}: entity rỗng (label={label})")

    return {
        "total":        len(data),
        "label_counts": label_counts,
        "issues":       issues
    }


def run_check(corpus_id: str):
    output_dir = Path("data/output") / corpus_id
    console.rule(f"[bold blue]Kiểm tra output: {corpus_id}")

    # ── _seg.tsv ────────────────────────────────────────────
    seg_path = output_dir / f"{corpus_id}_seg.tsv"
    console.print(f"\n[bold]📄 {seg_path.name}[/bold]")

    if not seg_path.exists():
        console.print(f"  [red]❌ File không tồn tại![/red]")
    else:
        seg_result = check_seg_tsv(seg_path)
        console.print(f"  ✅ Tổng số câu: [green]{seg_result['total']}[/green]")
        for issue in seg_result["issues"][:10]:
            console.print(f"  [yellow]⚠️  {issue}[/yellow]")
        if not seg_result["issues"]:
            console.print("  ✅ Không có lỗi format")

    # ── _ner.json ───────────────────────────────────────────
    ner_path = output_dir / f"{corpus_id}_ner.json"
    console.print(f"\n[bold]🏷️  {ner_path.name}[/bold]")

    if not ner_path.exists():
        console.print(f"  [red]❌ File không tồn tại![/red]")
    else:
        ner_result = check_ner_json(ner_path)
        console.print(f"  ✅ Tổng số câu: [green]{ner_result['total']}[/green]")

        # Bảng phân phối nhãn
        if ner_result["label_counts"]:
            table = Table(title="Phân phối nhãn NER", show_header=True)
            table.add_column("Nhãn", style="cyan")
            table.add_column("Số lượng", justify="right")
            table.add_column("Tỉ lệ", justify="right")

            total_ents = sum(ner_result["label_counts"].values())
            for label in sorted(VALID_LABELS):
                count = ner_result["label_counts"].get(label, 0)
                pct = f"{count/total_ents*100:.1f}%" if total_ents else "0%"
                table.add_row(label, str(count), pct)

            table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_ents}[/bold]", "100%")
            console.print(table)
        else:
            console.print("  [yellow]⚠️  Không có entity nào được nhận diện[/yellow]")

        for issue in ner_result["issues"][:10]:
            console.print(f"  [yellow]⚠️  {issue}[/yellow]")
        if not ner_result["issues"]:
            console.print("  ✅ Không có lỗi format")

    # ── _raw.txt ────────────────────────────────────────────
    raw_path = output_dir / f"{corpus_id}_raw.txt"
    console.print(f"\n[bold]📝 {raw_path.name}[/bold]")
    if raw_path.exists():
        size = raw_path.stat().st_size
        console.print(f"  ✅ Kích thước: [green]{size:,} bytes[/green]")
    else:
        console.print(f"  [red]❌ File không tồn tại![/red]")

    console.rule()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm tra chất lượng output HVH")
    parser.add_argument("--id", required=True, help="Mã tác phẩm (ví dụ: HVH_001_01)")
    args = parser.parse_args()
    run_check(args.id)
