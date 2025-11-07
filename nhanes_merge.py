import pandas as pd

# Load NHANES files
demo = pd.read_sas("P_DEMO.XPT", format="xport")
alq = pd.read_sas("P_ALQ.XPT", format="xport")
biopro = pd.read_sas("P_BIOPRO.XPT", format="xport")

# Merge on participant ID
df = demo.merge(alq, on="SEQN").merge(biopro, on="SEQN")

# Preview
print(df.head())
print(df.shape)

# Save to CSV so you can view it in Excel/Sheets
df.to_csv("nhanes_merged.csv", index=False)
