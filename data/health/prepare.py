import os
import tiktoken
import numpy as np
import pandas as pd

# Get the absolute path to the CSV, relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'chatdoctor_healthcaremagic.csv')

# Load the file
df = pd.read_csv(file_path)

#load dataset
#df = pd.read_csv('./chatdoctor_healthcaremagic.csv')
# Isolate just the 'output' text from the dataset
data = df['output'].str.cat(sep='\n')

# Split the data into training and validation sets
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has n tokens
# val.bin has n tokens
