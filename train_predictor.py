import torch
import joblib
from transformers import BertTokenizer, BertModel

# 모델 및 토크나이저 로드
model = joblib.load("rf_model.pkl")
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
bert_model = BertModel.from_pretrained("monologg/kobert")
bert_model.eval()

expected_traits = ["activity", "sociability", "care", "emotional_bond", "environment", "routine"]

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.pooler_output.squeeze(0).numpy().reshape(1, -1)

def predict_traits(input_text: str) -> dict:
    emb = get_embedding(input_text)
    prediction = model.predict(emb)[0]
    result = {
        trait: min(5, max(1, int(round(score))))
        for trait, score in zip(expected_traits, prediction)
    }
    return result

if __name__ == "__main__":
    print("성향 분석을 위한 문장을 입력하세요.")
    input_text = input(">> ")
    result = predict_traits(input_text)
    print("\n📊 예측된 성향 점수:")
    for trait, score in result.items():
        print(f"{trait}: {score}")