

import numpy as np

ENGINEERED_COLS = [
    "Sleep_Efficiency",
    "Work_Life_Balance_Ratio",
    "Screen_to_Sleep_Ratio",
    "Activity_Index",
    "Recovery_Score",
    "Caffeine_Sleep_Interaction",
]


def add_engineered_features(df):
    df = df.copy()

    # How much of a day's "awake budget" sleep actually covers (0-1 range-ish)
    df["Sleep_Efficiency"] = (df["Sleep_Hours"] / 9.0).clip(0, 1.3)

    # Work hours relative to combined rest+leisure time -> higher = worse balance
    rest_leisure = df["Weekend_Rest_Hours"] + df["Social_Interaction_Hours"] + 1e-3
    df["Work_Life_Balance_Ratio"] = df["Work_Hours"] / rest_leisure

    # Screen time relative to sleep -> higher = more digital overload per rest hour
    df["Screen_to_Sleep_Ratio"] = df["Screen_Time"] / (df["Sleep_Hours"] + 1e-3)

    # Composite of exercise + steps + meditation, min-max style scaled by rough caps
    df["Activity_Index"] = (
        (df["Exercise_Frequency"] / 7.0) * 0.4
        + (df["Daily_Steps"] / 15000.0).clip(0, 1) * 0.4
        + (df["Meditation_Minutes"] / 60.0).clip(0, 1) * 0.2
    )

    # How well the person recovers: sleep + weekend rest, penalized by stress
    df["Recovery_Score"] = (
        df["Sleep_Hours"] * 4 + df["Weekend_Rest_Hours"] * 2 - df["Stress_Score"] * 0.3
    )

    # Caffeine as a coping mechanism for poor sleep (interaction term)
    df["Caffeine_Sleep_Interaction"] = df["Caffeine_Intake"] * (9.0 - df["Sleep_Hours"]).clip(lower=0)

    return df


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("dataset/neurosync_dataset.csv")
    df = df.fillna(df.median(numeric_only=True))  # quick fill for standalone test run
    df = add_engineered_features(df)
    print(df[ENGINEERED_COLS].describe().round(2))
