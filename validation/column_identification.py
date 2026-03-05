import pandas as pd

mp = pd.read_csv("mediapipe_raw_features.csv")
of = pd.read_csv("openface_raw_features.csv")

print(mp.columns)
print(of.columns)