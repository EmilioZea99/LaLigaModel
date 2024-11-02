import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the dataset
csv_file_path = "la-liga-results-19952020/LaLiga_Matches.csv"
df = pd.read_csv(csv_file_path)

# 1. Distribution of Home Goals
plt.figure(figsize=(10, 6))
sns.histplot(df["FTHG"], bins=10, kde=True)
plt.title("Distribution of Full-Time Home Goals")
plt.xlabel("Goals Scored by Home Team")
plt.ylabel("Frequency")
plt.show()

# 2. Distribution of Away Goals
plt.figure(figsize=(10, 6))
sns.histplot(df["FTAG"], bins=10, kde=True, color="orange")
plt.title("Distribution of Full-Time Away Goals")
plt.xlabel("Goals Scored by Away Team")
plt.ylabel("Frequency")
plt.show()

# 3. Result Distribution (Home Win, Away Win, Draw)
plt.figure(figsize=(8, 5))
sns.countplot(x="FTR", data=df, palette="viridis")
plt.title("Full-Time Result Distribution")
plt.xlabel("Match Result")
plt.ylabel("Frequency")
plt.show()
