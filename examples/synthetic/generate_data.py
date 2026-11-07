"""Recreate a synthetic cohort with no participant data (NumPy seed 23)."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("destination", nargs="?", type=Path, default=Path(__file__).parent / "data" / "cohort.csv")
    target = ap.parse_args().destination
    if target.exists():
        raise FileExistsError("Choose a new destination; bundled data will not be overwritten")
    rng = np.random.default_rng(23)
    n = 80
    age = rng.normal(30, 5, n)
    latent = rng.normal(size=n)
    df = pd.DataFrame({
        "id": [f"synthetic_{i:03d}" for i in range(1, n + 1)],
        "age": age, "gender": np.arange(n) % 2,
        "L": 100 + latent + .1 * age + rng.normal(size=n),
        "R": 100 + latent + .15 * age + rng.normal(size=n),
        "Z": rng.normal(size=n),
        "y": latent + rng.normal(size=n), "z": rng.normal(size=n),
    })
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False, float_format="%.12g")
    print(target)


if __name__ == "__main__":
    main()
