from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import pandas as pd
import json

# Prepare Datasets
print("[INFO] Download Datasets...")
klue_nli = load_dataset('klue', 'nli')
valid_df = klue_nli['validation'].to_pandas()
valid_df_300 = valid_df[:300]

# Prepare Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("Huffon/klue-roberta-base-nli")
model = AutoModelForSequenceClassification.from_pretrained("Huffon/klue-roberta-base-nli")

# Prepare Pipeline
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer = tokenizer,
    #return_all_scores=True,
)

# Start Inference
print("[INFO] Inference...")   
predictions = []
for i in tqdm(range(len(valid_df_300))):
    input = str(valid_df_300['premise'][i]) + tokenizer.sep_token + str(valid_df_300['hypothesis'][i])
    prediction = classifier(input)[0]['label']
    predictions.append(prediction)

# Convert Label (str -> int)
for i in tqdm(range(len(predictions))):
    predictions[i] = predictions[i].replace('CONTRADICTION', "2")
    predictions[i] = predictions[i].replace('NEUTRAL', "1")
    predictions[i] = predictions[i].replace('ENTAILMENT', "0")
predictions = list(map(int, predictions))

# Save to json
with open('./huggingface.json', "w") as f:
    json.dump(predictions, f)#, indent=4)
print("[INFO] Done!")
