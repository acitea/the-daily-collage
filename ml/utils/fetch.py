import hopsworks
import pandas as pd
from sklearn.model_selection import train_test_split

# Fetch from Hopsworks
project = hopsworks.login(api_key_value="YOUR_KEY", project="daily_collage", engine="python")
fs = project.get_feature_store()
fg = fs.get_feature_group(name="headline_labels", version=1)
df = fg.read()

# Split train/val
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Save
train_df.to_parquet("data/train.parquet")
val_df.to_parquet("data/val.parquet")

print(f"✓ Fetched {len(train_df)} train and {len(val_df)} val samples from Hopsworks")