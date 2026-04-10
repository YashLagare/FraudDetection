import pandas as pd
import os

filename = 'PS_20174392719_1491204439457_log.csv'
print(f"Reading {filename}...")
df = pd.read_csv(filename)

print("Splitting into fraud and non-fraud...")
fraud = df[df['isFraud'] == 1]
non_fraud = df[df['isFraud'] == 0]

# Sample down non-fraud to about 1,000,000 rows so that total size is safely < 100MB (around 80MB expected)
num_samples = 1000000
print(f"Sampling {num_samples} out of {len(non_fraud)} non-fraud rows...")
if len(non_fraud) > num_samples:
    non_fraud_sampled = non_fraud.sample(n=num_samples, random_state=42)
else:
    non_fraud_sampled = non_fraud

print("Combining and shuffling...")
df_sampled = pd.concat([fraud, non_fraud_sampled]).sample(frac=1, random_state=42)

out_filename = 'PS_20174392719_1491204439457_log_small.csv'
print(f"Saving to {out_filename}...")
df_sampled.to_csv(out_filename, index=False)

print(f"Done! Original size: {os.path.getsize(filename)/(1024*1024):.2f} MB")
print(f"New size: {os.path.getsize(out_filename)/(1024*1024):.2f} MB")
