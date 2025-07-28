from datasets import load_dataset

ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
train_ds = ds["train"]
df = train_ds.to_pandas()
print(df.head())
df.to_csv("chatdoctor_healthcaremagic.csv", index=False)