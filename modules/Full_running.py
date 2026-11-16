import os
import sys
import time


REPO_ROOT = r"C:\Users\Peter\Downloads\barca-hackathon-main\barca-hackathon-main"   # <-- EDIT THIS

DEADBALL_DIR = os.path.join(REPO_ROOT, "modules", "deadball")
POSSESSION_DIR = os.path.join(REPO_ROOT, "modules", "possession")   

TRIM_MIN = 1
TRIM_MAX = 500
COARSE_STEP = 2    
FINE_STEP = 1       
FINE_WINDOW = 20    

N_FRAMES = 199319
GT_PATH = os.path.join(REPO_ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
OUT_XLSX = "final_trim_sweep.xlsx"

# =========================================================================== #
#  [SETUP] Check every required file exists before importing anything
# =========================================================================== #

REQUIRED = [
    ("games.py", "games.py"),
    ("modules/deadball/dead_ball.py", os.path.join("modules", "deadball", "dead_ball.py")),
    ("starter/load_smart_tagging.py", os.path.join("starter", "load_smart_tagging.py")),
    ("possession.py", os.path.join("modules", "possession", "possession.py")),
    ("atd_arrays.py", os.path.join("modules", "possession", "atd_arrays.py")),
    ("players_model.py", os.path.join("modules", "possession", "players_model.py")),
    ("evaluation/possession_metrics.py", os.path.join("evaluation", "possession_metrics.py")),
    ("evaluation/gt/BAR-ATH_possession_gt_public.csv",
     os.path.join("evaluation", "gt", "BAR-ATH_possession_gt_public.csv")),
]


def check_setup():
    missing = [(label, os.path.join(REPO_ROOT, rel)) for label, rel in REQUIRED
               if not os.path.exists(os.path.join(REPO_ROOT, rel))]
    if missing:
        lines = ["Missing {} required item(s):".format(len(missing))]
        for label, full in missing:
            lines.append("  [MISSING] {} (expected at: {})".format(label, full))
        lines.append("")
        lines.append("If 'possession.py' / 'atd_arrays.py' / 'players_model.py' are missing, "
                      "check the real folder name in your repo and update POSSESSION_DIR above.")
        raise SystemExit("\n".join(lines))
    print("[SETUP] All required files found.\n")


check_setup()

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "evaluation"))
sys.path.insert(0, DEADBALL_DIR)
sys.path.insert(0, POSSESSION_DIR)

import numpy as np
import pandas as pd

from dead_ball import detect_dead_ball                      # Part 1
from starter.load_smart_tagging import load as load_tagging
import possession as ps                                     # Part 2 
import possession_metrics as pm                             

# =========================================================================== #
#  [SETUP] Everything that does NOT depend on trim_frames
# =========================================================================== #

print("=" * 70)
print("[SETUP] Loading Smart Tagging events ...")
events = load_tagging()
set_pieces = [e for e in events if e["code"] == "SET PIECES"]

print("[SETUP] Loading possession signals -- parses raw tracking and trains")
t0 = time.perf_counter()
S = ps.load_signals()
print("        done in {:.0f}s".format(time.perf_counter() - t0))

own_stoppage = ps.stoppage_mask(S)  

print("[SETUP] Loading ground truth ...")
gt_codes = pm.label_codes(pm.read_label_csv(GT_PATH))
print("=" * 70 + "\n")


def evaluate_trim(trim_frames):
    is_dead = detect_dead_ball(set_pieces, n_frames=N_FRAMES, trim_frames=trim_frames)
    dead = (own_stoppage | is_dead) & S["graded"]

    team = ps.decode_hmm(S, dead)
    labels = ps.to_labels(team, dead, S)

    pred_codes = pm.label_codes(labels)
    result = pm.evaluate(gt_codes, pred_codes, bootstrap_n=0)

    return {
        "trim_frames": trim_frames,
        "trim_seconds": round(trim_frames / 25.0, 2),
        "spellF1": result["spell_f1_mean"],
        "macroF1": result["macro_f1"],
        "n_pred_spells": result["n_pred_spells"],
        "n_gt_spells": result["n_gt_spells"],
    }, labels


def run_pass(trims, pass_name):
    print("-" * 70)
    print("[{}] sweeping {} trim value(s): {} .. {}".format(
        pass_name, len(trims), trims[0], trims[-1]))
    print("-" * 70)
    rows = []
    t0 = time.perf_counter()
    for i, trim in enumerate(trims, start=1):
        row, _ = evaluate_trim(trim)
        row["pass"] = pass_name
        rows.append(row)
        if i % 20 == 0 or i == len(trims):
            print("  {}/{} done ({:.0f}s elapsed)".format(i, len(trims), time.perf_counter() - t0))
    print()
    return rows


# =========================================================================== #
#  [PASS 1 -- COARSE] wide range, coarse step
# =========================================================================== #

coarse_trims = list(range(TRIM_MIN, TRIM_MAX + 1, COARSE_STEP))
coarse_rows = run_pass(coarse_trims, "PASS 1 - COARSE")

peak_trim = max(coarse_rows, key=lambda r: (r["spellF1"], r["macroF1"]))["trim_frames"]
print("[PASS 1 - COARSE] peak so far: trim_frames={}\n".format(peak_trim))

# =========================================================================== #
#  [PASS 2 -- FINE] narrow window around the coarse peak, fine step
# =========================================================================== #

fine_lo = max(TRIM_MIN, peak_trim - FINE_WINDOW)
fine_hi = min(TRIM_MAX, peak_trim + FINE_WINDOW)
fine_trims = [t for t in range(fine_lo, fine_hi + 1, FINE_STEP) if t not in coarse_trims]
fine_rows = run_pass(fine_trims, "PASS 2 - FINE") if fine_trims else []

# =========================================================================== #
#  [RESULT] combine both passes, rank, export, and write the best submission
# =========================================================================== #

all_rows = coarse_rows + fine_rows
df = pd.DataFrame(all_rows)
df = df.sort_values(["spellF1", "macroF1"], ascending=[False, False]).reset_index(drop=True)
df.insert(0, "rank", np.arange(1, len(df) + 1))

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Results", index=False)

from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill

wb = load_workbook(OUT_XLSX)
ws = wb["Results"]
header_font = Font(name="Arial", bold=True)
best_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
for cell in ws[1]:
    cell.font = header_font
ws.freeze_panes = "A2"
for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
    if row[0].value == 1:
        for cell in row:
            cell.fill = best_fill
for col_cells in ws.columns:
    width = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells) + 2
    ws.column_dimensions[col_cells[0].column_letter].width = width
wb.save(OUT_XLSX)

best_trim = int(df.iloc[0]["trim_frames"])
print("=" * 70)
print("[RESULT] Best trim_frames = {}  (spellF1 = {:.4f}, macroF1 = {:.4f})".format(
    best_trim, df.iloc[0]["spellF1"], df.iloc[0]["macroF1"]))
print("[RESULT] Saved ranked results: {}".format(OUT_XLSX))
print("=" * 70)

print("\nTop 10 by spellF1:")
print(df.head(10).to_string(index=False))


print("\n Writing final submission for the best trim found ...")
_, best_labels = evaluate_trim(best_trim)
out_submission = os.path.join(
    REPO_ROOT, "evaluation", "results", "final_trim{}.csv".format(best_trim))
os.makedirs(os.path.dirname(out_submission), exist_ok=True)
with open(out_submission, "w") as f:
    f.write("frame,label\n")
    for i, label in enumerate(best_labels):
        f.write("{},{}\n".format(i, label))
print("wrote {}".format(out_submission))