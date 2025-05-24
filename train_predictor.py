import torch
import joblib
from transformers import BertTokenizer, BertModel

# 모델과 토크나이저 로드
model = joblib.load("rf_model.pkl")
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
bert_model = BertModel.from_pretrained("monologg/kobert")
bert_model.eval()

expected_traits = ["activity", "sociability", "care", "emotional_bond", "environment", "routine"]

# 여러 문장의 평균 임베딩 계산
def get_mean_embedding(texts: list[str]) -> torch.Tensor:
    embs = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        with torch.no_grad():
            output = bert_model(**inputs)
        embs.append(output.pooler_output.squeeze(0))
    return torch.stack(embs).mean(dim=0).numpy().reshape(1, -1)

# 성향 점수 예측 함수
def predict_traits(qna_list: list[str]) -> dict:
    emb = get_mean_embedding(qna_list)
    prediction = model.predict(emb)[0]
    result = {
        trait: min(5, max(1, int(round(score))))
        for trait, score in zip(expected_traits, prediction)
    }
    return result

# 터미널 테스트 실행용
if __name__ == "__main__":
    print("10개의 질문+응답을 입력하세요 (예: activity: 어떤 활동을 좋아하세요? [SEP] 산책을 자주 합니다.)")

    qna_list = []
    for i in range(10):
        qna = input(f"{i+1} >> ")
        qna_list.append(qna)

    predictions = predict_traits(qna_list)

    print("\n📊 예측된 성향 점수:")
    for trait, score in predictions.items():
        print(f"{trait}: {score}")