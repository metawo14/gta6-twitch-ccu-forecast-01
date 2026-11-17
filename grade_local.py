"""Local dev grader: runs main.py (and optionally the starter) on the practice cases and scores
the output against the published SullyGnome release-day curves.

    python3 grade_local.py            # grade main.py
    python3 grade_local.py --baseline # grade the kit's starter (house baseline) instead

Score: mean pinball loss over hours 2..24 and the five levels, in log space, y < 1 counted as 1.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

KIT = Path("/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit")
ENTRY = Path(__file__).resolve().parent
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

ANSWERS = {
    "p00": Path("/tmp/sg-wolverine.json"),
    "p01": Path("/tmp/sg-rdr2.json"),
    "p02": Path("/tmp/sg-gow.json"),
    "p03": Path("/tmp/sg-spiderman2.json"),
}
EXPECTED_P1 = {"p00": 23357, "p01": 116550, "p02": 269517, "p03": 119087}
QUESTIONS = {"p00": "2026-09-15", "p01": "2018-10-26", "p02": "2022-11-09", "p03": "2023-10-20"}


def answer_curve(cid):
    """Hours 2..24 of the release day from the 'Average viewers' dataset (not the trend one)."""
    data = json.loads(ANSWERS[cid].read_text())
    labels = data["data"]["labels"]
    dataset = next(d for d in data["data"]["datasets"] if d["label"] == "Average viewers")
    by_label = dict(zip(labels, dataset["data"]))
    day = QUESTIONS[cid]
    hour1 = by_label[f"{day} 00:00"]
    if int(hour1) != EXPECTED_P1[cid]:
        raise SystemExit(f"{cid}: hour-1 answer {hour1} != known {EXPECTED_P1[cid]}")
    curve = {}
    for h in range(2, 25):
        key = f"{day} {(h - 1) % 24:02d}:00"
        if key not in by_label or by_label[key] is None:
            raise SystemExit(f"{cid}: no answer for hour {h} ({key})")
        curve[h] = by_label[key]
    return curve


def pinball(tau, y, q):
    d = math.log(max(1.0, y)) - math.log(max(1e-12, q))
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def run_case(cid, program_dir, program="main.py"):
    with tempfile.TemporaryDirectory() as tmp:
        indir = Path(tmp) / "in"
        outdir = Path(tmp) / "out"
        indir.mkdir()
        outdir.mkdir()
        shutil.copy(KIT / "input" / f"{cid}.in", indir / f"{cid}.in")
        env = dict(os.environ, MAKEFUN_INPUT_DIR=str(indir), MAKEFUN_OUTPUT_DIR=str(outdir))
        subprocess.run(
            [sys.executable, str(Path(program_dir) / program)],
            check=True, env=env, cwd=tmp,
        )
        return json.loads((outdir / f"{cid}.out").read_text())


def score(forecast, curve):
    hours = {row["hour"]: row["q"] for row in forecast["hours"]}
    items = []
    for h in range(2, 25):
        for i, tau in enumerate(LEVELS):
            items.append(pinball(tau, curve[h], hours[h][i]))
    return sum(items) / len(items)


def validate(forecast, cid):
    assert list(forecast.keys()) == ["hours"], f"{cid}: keys {list(forecast.keys())}"
    rows = forecast["hours"]
    assert len(rows) == 23, f"{cid}: {len(rows)} rows"
    seen = set()
    for row in rows:
        assert set(row.keys()) == {"hour", "q"}, f"{cid}: row keys {row.keys()}"
        h = row["hour"]
        assert h not in seen, f"{cid}: duplicate hour {h}"
        seen.add(h)
        q = row["q"]
        assert len(q) == 5, f"{cid}: {len(q)} quantiles"
        for v in q:
            assert isinstance(v, (int, float)) and math.isfinite(v), f"{cid}: bad q {v}"
            assert 0 < v <= 1e9, f"{cid}: q out of range {v}"
        assert all(q[i] <= q[i + 1] for i in range(4)), f"{cid}: not monotone {q}"
    assert seen == set(range(2, 25)), f"{cid}: hours {sorted(seen)}"


def main():
    baseline = "--baseline" in sys.argv
    program_dir = KIT / "starter" if baseline else ENTRY
    losses = {}
    for cid in sorted(ANSWERS):
        forecast = run_case(cid, program_dir)
        validate(forecast, cid)
        losses[cid] = score(forecast, answer_curve(cid))
        print(f"{cid}: {losses[cid]:.6f}")
    mean = sum(losses.values()) / len(losses)
    label = "baseline" if baseline else "entry"
    print(f"{label} mean: {mean:.6f}")


if __name__ == "__main__":
    main()
