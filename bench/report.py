from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    results_path = Path(__file__).parent / "out" / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    # Group by workload then threshold
    grouped: dict[str, list[dict]] = {}
    for r in data:
        grouped.setdefault(r["workload"], []).append(r)

    out = []
    out.append("# Benchmark Results\n")

    for workload, rows in grouped.items():
        rows = sorted(rows, key=lambda x: x["threshold"])
        out.append(f"## {workload}\n")
        out.append("| threshold | hit_rate | unsafe_hit_rate | token_savings | cost_savings | p50_ms | p95_ms |")
        out.append("|---:|---:|---:|---:|---:|---:|---:|")

        for r in rows:
            s = r["summary"]
            out.append(
                f"| {r['threshold']:.2f} "
                f"| {s['hit_rate']*100:.1f}% "
                f"| {s['unsafe_hit_rate']*100:.2f}% "
                f"| {s['token_savings']*100:.1f}% "
                f"| {s['cost_savings']*100:.1f}% "
                f"| {s['latency_p50_ms']:.1f} "
                f"| {s['latency_p95_ms']:.1f} |"
            )
        out.append("")

    report_path = Path(__file__).parent / "out" / "REPORT.md"
    report_path.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
