from datasets import load_dataset
import openai
import pandas as pd
from tqdm import tqdm
import json
from time import sleep

# Prepare Datasets
print("[INFO] Download Datasets...")
klue_nli = load_dataset('klue', 'nli')
valid_df = klue_nli['validation'].to_pandas()
valid_df_300 = valid_df[:300]

# Set API Key
##MY_API_KEY = ""
openai.api_key = MY_API_KEY

# Set Model
model = 'gpt-3.5-turbo'

# Start Inference
print("[INFO] Inference...")   
predictions = []
sleep_count = 0
for i in tqdm(range(len(valid_df_300))):
    sleep_count +=1
    query = '앞선 문장에 대해서 다음 문장이 논리적으로 수반되는지, 모순적인지, 중립적인지를 구분하고 싶어. [SEP]으로 구분되어 있는 다음의 두 문장에 대해서, "중립", "모순", "수반"의 셋 중 하나로 분류해줘.' + str(valid_df_300['premise'][i]) + "[SEP]" + str(valid_df_300['hypothesis'][i]) + " -> "
    messages = [
        {"role": "system", "content": "당신은 두 문장 사이의 관계를 잘 포착해서, 한국어로 답할 수 있습니다."},
        {"role": "user", "content": query}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )
    answer = response['choices'][0]['message']['content']
    predictions.append(answer)
    if sleep_count % 30 == 0:
        sleep(120)    

# PostProcessing 
for i in tqdm(range(len(predictions))):
    if "수반" in str(predictions[i]):
        predictions[i] = "0"
    elif "모순" in str(predictions[i]):
        predictions[i] = "2"
    elif "중립" in str(predictions[i]):
        predictions[i] = "1"
predictions = list(map(int, predictions))

# Save to json
with open('./chatgpt.json', "w") as f:
    json.dump(predictions, f)
print("[INFO] Done!")