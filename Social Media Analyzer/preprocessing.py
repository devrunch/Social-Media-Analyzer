# import pandas as pd
# import numpy as np

# # Load CSV
# df = pd.read_csv("data/projectML_augmented_final.csv")

# # Convert Timestamp to datetime
# df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# # Add Weekday
# df['Weekday'] = df['Timestamp'].dt.day_name()

# # Define Time Period
# def get_time_period(hour):
#     if 5 <= hour < 12:
#         return 'Morning'
#     elif 12 <= hour < 17:
#         return 'Afternoon'
#     elif 17 <= hour < 21:
#         return 'Evening'
#     else:
#         return 'Night'

# df['TimePeriod'] = df['Timestamp'].dt.hour.apply(get_time_period)

# # Add Random/Fake Likes, Comments, Shares
# np.random.seed(42)
# df['Likes'] = np.random.randint(50, 1000, size=len(df))
# df['Comments'] = np.random.randint(5, 200, size=len(df))
# df['Shares'] = np.random.randint(1, 100, size=len(df))

# # Total Engagement
# df['Engagement'] = df['Likes'] + df['Comments'] + df['Shares']

# # Save updated dataset
# df.to_csv("projectML_augmented_with_engagement.csv", index=False)

import pandas as pd
import numpy as np
import re
from datetime import datetime

# Load your original dataset
df = pd.read_csv("projectML_augmented_with_engagement.csv")

# 1. Text-based features
df["TextLength"] = df["Text"].apply(lambda x: len(str(x)))
df["WordCount"] = df["Text"].apply(lambda x: len(str(x).split()))
df["HashtagCount"] = df["Text"].apply(lambda x: len(re.findall(r"#\w+", str(x))))
df["MentionCount"] = df["Text"].apply(lambda x: len(re.findall(r"@\w+", str(x))))
df["EmojiCount"] = df["Text"].apply(lambda x: len(re.findall(r"[^\w\s,]", str(x))))
df["ExclamationCount"] = df["Text"].apply(lambda x: str(x).count("!"))

# 2. Timestamp-based features
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Hour"] = df["Timestamp"].dt.hour
df["IsWeekend"] = df["Timestamp"].dt.weekday >= 5

# 3. Engagement ratios
df["EngagementPerWord"] = df["Engagement"] / (df["WordCount"] + 1)
df["LikeRatio"] = df["Likes"] / (df["Engagement"] + 1)
df["CommentRatio"] = df["Comments"] / (df["Engagement"] + 1)
df["ShareRatio"] = df["Shares"] / (df["Engagement"] + 1)

# Save the updated dataset
df.to_csv("feature_engineered_dataset.csv", index=False, encoding='utf-8')
print("Saved: feature_engineered_dataset.csv")
