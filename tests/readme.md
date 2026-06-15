Here are the 3 files. Quick summary of what each does and where it lives:

**`generate_factorial_jobs.py`** — goes in your Unreal project scripts folder, run in the UE Python console. Set `SCENE_CONFIG` at the top with your level/sequence paths. Creates 82 MRQ jobs (81 factorial + 1 reference).

**`run_factorial_metrics.py`** — goes in `tests/`. Run as:
```bash
python run_factorial_metrics.py --scene quarry-rocksonly
python run_factorial_metrics.py --scene quarry-rocksonly --resume  # if interrupted
```
Outputs `scores.json` and `scores.csv` per scene. It caches the CGVQM model weights in memory across all 81 combos so you're not reloading for every render.

**`analyze_independence.py`** — also in `tests/`. Run after scoring:
```bash
python analyze_independence.py --scene quarry-rocksonly
python analyze_independence.py --scene s1 s2 s3 --pool  # combined analysis
```
Fits the two models (main effects vs +interactions), runs the F-test, and prints an ANOVA table with η² effect sizes per term — exactly what reviewers will want to see. Saves a `independence_report.txt` per scene.
