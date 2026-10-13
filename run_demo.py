"""
Single entry point for evaluating this project.

    python run_demo.py

Does everything in order, with no prior setup beyond installing the four
packages in requirements.txt:

    1. Checks the Python version and that dependencies are importable
    2. Verifies corpus coverage across all intents and languages
    3. Trains the model if no saved model is present (skips if one exists)
    4. Reports honest, out-of-distribution accuracy
    5. Runs a live demonstration on real code-mixed messages

Nothing is written outside this folder. Total run time is about three minutes
on a first run, or about twenty seconds if the model has already been trained.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "ndr_model.joblib")
PY = sys.executable          # reuse whichever interpreter launched this file


def rule(char: str = "=") -> None:
    print(char * 74)


def banner(n: str, title: str) -> None:
    print()
    rule()
    print(f"  STEP {n}   {title}")
    rule()


def run(script: str, *args: str) -> bool:
    """Run a sibling script with the same interpreter. Returns success."""
    proc = subprocess.run([PY, os.path.join(HERE, script), *args], cwd=HERE)
    return proc.returncode == 0


def main() -> int:
    t0 = time.time()

    print()
    rule()
    print("  NDR RESOLUTION SYSTEM  \u2014  full demonstration")
    print("  An NLP system that reads customer replies to failed deliveries")
    print("  and recommends what the operations team should do.")
    rule()
    print(f"  Python     : {sys.version.split()[0]}")
    print(f"  Interpreter: {PY}")
    print(f"  Folder     : {HERE}")

    # ---------------------------------------------------------------- 0
    if sys.version_info < (3, 10):
        print("\n  ERROR: Python 3.10 or newer is required "
              f"(found {sys.version.split()[0]}).")
        print("  Install a newer Python from https://www.python.org/downloads/")
        return 1

    missing = []
    for pkg in ("sklearn", "pandas", "numpy", "joblib"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append("scikit-learn" if pkg == "sklearn" else pkg)
    if missing:
        print(f"\n  ERROR: missing package(s): {', '.join(missing)}")
        print(f"  Install them with:\n\n      {PY} -m pip install -r requirements.txt\n")
        return 1

    # ---------------------------------------------------------------- 1
    banner("1 / 4", "Checking the installation")
    if not run("verify_setup.py"):
        print("\n  Setup check failed. See the messages above.")
        return 1

    # ---------------------------------------------------------------- 2
    banner("2 / 4", "Verifying the synthetic corpus covers every category")
    if not run("verify_coverage.py"):
        print("\n  Coverage check failed. See the messages above.")
        return 1

    # ---------------------------------------------------------------- 3
    if os.path.exists(MODEL):
        banner("3 / 4", "Trained model found \u2014 skipping training")
        print(f"  Using {os.path.basename(MODEL)}")
        print(f"  Delete that file and re-run to retrain from scratch "
              f"(takes ~2\u20133 minutes).")
    else:
        banner("3 / 4", "Training the model (about 2\u20133 minutes)")
        print("  Note: the accuracy printed below is around 0.33. That is the")
        print("  correct, honest figure \u2014 see step 4 for why.\n")
        if not run("train.py"):
            print("\n  Training failed. See the messages above.")
            return 1

    # ---------------------------------------------------------------- 4
    banner("4 / 4", "Evaluating on messages the model has never seen")
    if not run("eval_challenge.py"):
        print("\n  Evaluation failed. See the messages above.")
        return 1

    # ---------------------------------------------------------------- demo
    print()
    rule()
    print("  LIVE DEMONSTRATION")
    rule()
    if not run("predict.py", "--demo"):
        return 1

    # ---------------------------------------------------------------- close
    print()
    rule()
    print(f"  Finished in {time.time() - t0:.0f} seconds.")
    rule()
    print("""
  HOW TO READ THE RESULTS

  Two accuracy figures appear above, and they mean different things.

    ~0.99   Reported by train.py when messages are split at random. This
            is inflated: the model is tested on rewordings of sentences it
            already trained on. It is shown only for comparison.

    ~0.33   Reported when whole reasons are held out, so the model is
            tested on ways of saying things it has never seen. This is the
            figure the project reports.

  Accuracy is not the main measure. Because the system escalates anything
  it is unsure about, only about 2.6% of messages receive a wrong action
  with no human involved. Cancellations and returns are never automated,
  regardless of confidence.

  TRY YOUR OWN MESSAGE

      {py} predict.py --text "kal shaam bhej dena ghar pe koi nahi hai"
      {py} predict.py --text "nobody came, i waited all day"
      {py} predict.py --csv replies.csv --out resolved.csv

  DOCUMENTATION

      PROJECT_REPORT.md   full written report
      PROJECT_GUIDE.md    plain-language explanation
      DATA_PLAN.md        where the data comes from
""".format(py=os.path.basename(PY)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
