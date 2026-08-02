"""
NeuroSync AI - Synthetic Dataset Generator
--------------------------------------------
Generates a realistic, correlated lifestyle/burnout dataset with 100,000+ rows
and 20+ features, with a balanced 3-class Burnout_Risk target
(Low / Medium / High).

Run:
    python generate_dataset.py
Output:
    dataset/neurosync_dataset.csv
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_ROWS = 120_000

rng = np.random.default_rng(RANDOM_SEED)


def clip(arr, low, high):
    return np.clip(arr, low, high)


def generate_dataset(n=N_ROWS):
    # ---------------------------------------------------------------
    # 1. Demographics
    # ---------------------------------------------------------------
    age = rng.integers(18, 66, size=n)

    gender = rng.choice(
        ["Male", "Female", "Other"], size=n, p=[0.48, 0.48, 0.04]
    )

    occupation = rng.choice(
        ["IT/Software", "Healthcare", "Education", "Business/Finance",
         "Student", "Freelancer", "Other"],
        size=n,
        p=[0.22, 0.14, 0.12, 0.16, 0.18, 0.10, 0.08],
    )

    # Base workload multiplier per occupation (drives work hours / screen time)
    occ_load = {
        "IT/Software": 1.25,
        "Healthcare": 1.30,
        "Education": 1.05,
        "Business/Finance": 1.20,
        "Student": 0.85,
        "Freelancer": 1.00,
        "Other": 0.95,
    }
    load_factor = np.array([occ_load[o] for o in occupation])

    # ---------------------------------------------------------------
    # 2. Work / lifestyle behavior (correlated with occupation load)
    # ---------------------------------------------------------------
    work_hours = clip(
        rng.normal(7.5, 1.8, n) * load_factor, 2, 16
    )

    screen_time = clip(
        rng.normal(5.5, 2.0, n) * (0.6 + 0.4 * load_factor)
        + 0.25 * (work_hours - 7.5),
        1, 16,
    )

    # Sleep drops as work hours & screen time rise
    sleep_hours = clip(
        8.2
        - 0.18 * (work_hours - 7.5)
        - 0.10 * (screen_time - 5.5)
        + rng.normal(0, 0.8, n),
        3, 10,
    )

    exercise_frequency = clip(
        rng.poisson(3, n) - 0.15 * (work_hours - 7.5) + rng.normal(0, 0.5, n),
        0, 7,
    ).round()

    daily_steps = clip(
        3000 + exercise_frequency * 900 + rng.normal(0, 1500, n) - 60 * (screen_time - 5.5),
        500, 20000,
    ).round()

    water_intake = clip(
        1.5 + 0.15 * exercise_frequency + rng.normal(0, 0.5, n), 0.3, 5.0
    )

    caffeine_intake = clip(
        1.5 + 0.3 * (work_hours - 7.5) + rng.normal(0, 1.0, n), 0, 10
    ).round()

    social_interaction_hours = clip(
        3.0 - 0.15 * (screen_time - 5.5) + rng.normal(0, 1.0, n), 0, 10
    )

    meditation_minutes = clip(
        rng.exponential(8, n) - 0.5 * (work_hours - 7.5) + rng.normal(0, 2, n),
        0, 90,
    ).round()

    weekend_rest_hours = clip(
        rng.normal(9, 2.0, n) - 0.1 * (work_hours - 7.5), 2, 16
    )

    # ---------------------------------------------------------------
    # 3. Physiological indicators
    # ---------------------------------------------------------------
    bmi = clip(
        23 + 0.05 * (age - 40) - 0.6 * exercise_frequency
        - 0.3 * water_intake + rng.normal(0, 3.0, n),
        15, 45,
    )

    heart_rate = clip(
        70 + 0.4 * (bmi - 23) - 1.2 * exercise_frequency
        + 0.5 * caffeine_intake + rng.normal(0, 6, n),
        50, 130,
    ).round()

    bp_systolic = clip(
        112 + 0.3 * (age - 40) + 0.3 * (bmi - 23) + rng.normal(0, 8, n),
        90, 180,
    ).round()
    bp_diastolic = clip(
        bp_systolic * 0.62 + rng.normal(0, 4, n), 55, 115
    ).round()

    # ---------------------------------------------------------------
    # 4. Psychological scores (0-100)
    # ---------------------------------------------------------------
    stress_score = clip(
        50
        + 2.2 * (work_hours - 7.5)
        + 1.8 * (screen_time - 5.5)
        - 3.0 * (sleep_hours - 7.5)
        - 1.2 * exercise_frequency
        - 0.3 * meditation_minutes
        + rng.normal(0, 8, n),
        0, 100,
    )

    mood_score = clip(
        60
        - 0.35 * (stress_score - 50)
        + 1.0 * exercise_frequency
        + 0.8 * social_interaction_hours
        + 0.15 * meditation_minutes
        + rng.normal(0, 8, n),
        0, 100,
    )

    productivity_score = clip(
        65
        - 0.30 * (stress_score - 50)
        + 1.3 * (sleep_hours - 7.5)
        - 0.5 * (screen_time - 5.5)
        + 0.4 * exercise_frequency
        + rng.normal(0, 8, n),
        0, 100,
    )

    wellness_score = clip(
        0.35 * mood_score
        + 0.30 * (100 - stress_score)
        + 0.20 * (sleep_hours / 10 * 100)
        + 0.15 * (exercise_frequency / 7 * 100)
        + rng.normal(0, 4, n),
        0, 100,
    )

    # ---------------------------------------------------------------
    # 5. Burnout risk score -> balanced 3-class target
    # ---------------------------------------------------------------
    burnout_raw = (
        0.30 * stress_score
        - 0.20 * wellness_score
        - 0.15 * (sleep_hours / 10 * 100)
        + 0.15 * (work_hours / 16 * 100)
        + 0.10 * (screen_time / 16 * 100)
        - 0.10 * (exercise_frequency / 7 * 100)
        + rng.normal(0, 6, n)
    )

    # Quantile-based binning guarantees a balanced target distribution
    q1, q2 = np.quantile(burnout_raw, [1 / 3, 2 / 3])
    burnout_risk = np.select(
        [burnout_raw <= q1, burnout_raw <= q2],
        ["Low", "Medium"],
        default="High",
    )

    df = pd.DataFrame(
        {
            "Age": age,
            "Gender": gender,
            "Occupation": occupation,
            "Work_Hours": work_hours.round(2),
            "Sleep_Hours": sleep_hours.round(2),
            "Screen_Time": screen_time.round(2),
            "Exercise_Frequency": exercise_frequency.astype(int),
            "Daily_Steps": daily_steps.astype(int),
            "Water_Intake": water_intake.round(2),
            "Caffeine_Intake": caffeine_intake.astype(int),
            "BMI": bmi.round(2),
            "Heart_Rate": heart_rate.astype(int),
            "BP_Systolic": bp_systolic.astype(int),
            "BP_Diastolic": bp_diastolic.astype(int),
            "Stress_Score": stress_score.round(2),
            "Mood_Score": mood_score.round(2),
            "Social_Interaction_Hours": social_interaction_hours.round(2),
            "Meditation_Minutes": meditation_minutes.astype(int),
            "Weekend_Rest_Hours": weekend_rest_hours.round(2),
            "Productivity_Score": productivity_score.round(2),
            "Wellness_Score": wellness_score.round(2),
            "Burnout_Risk": burnout_risk,
        }
    )

    # ---------------------------------------------------------------
    # 6. Inject realistic imperfections (missing values + duplicates)
    #    so the preprocessing pipeline has real work to do.
    # ---------------------------------------------------------------
    missing_cols = ["Sleep_Hours", "Water_Intake", "Meditation_Minutes", "BMI", "Social_Interaction_Hours"]
    for col in missing_cols:
        mask = rng.random(n) < 0.015  # ~1.5% missing per column
        df.loc[mask, col] = np.nan

    # duplicate ~0.3% of rows
    dup_idx = rng.choice(df.index, size=int(n * 0.003), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # shuffle rows
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    return df


if __name__ == "__main__":
    dataset = generate_dataset()
    out_path = "neurosync_dataset.csv"
    dataset.to_csv(out_path, index=False)

    print(f"Dataset shape: {dataset.shape}")
    print(f"Missing values:\n{dataset.isna().sum()[dataset.isna().sum() > 0]}")
    print(f"Duplicate rows: {dataset.duplicated().sum()}")
    print(f"\nBurnout_Risk distribution:\n{dataset['Burnout_Risk'].value_counts(normalize=True)}")
    print(f"\nSaved to {out_path}")
