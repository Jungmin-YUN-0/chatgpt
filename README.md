**(chatgpt-practice)**

## [overview]

huggingface transformers 라이브러리와 ChatGPT API를 각각 활용해 KLUE NLI의 validation set을 inference 하기

### [huggingface.py]

https://huggingface.co/models에서 모델 선정 후, KLUE NLI 데이터셋 validation set에 대해 inference 수행하고 결과 저장(→ huggingface.json)

### [chatgpt.py]

ChatGPT API를 활용해 prompt 디자인 후, KLUE NLI validation set에 대해 inference 수행하고 결과 저장(→ chatgpt.json)
