from train_predictor import predict_traits
from train_knn_model import recommend, load_data_from_db

def main():
    
    # 1. 사용자 입력 받기
    print("성향 분석을 위한 문장을 입력하세요.")
    input_text = input(">> ")

    # 2. KoBERT + RF로 성향 점수 예측
    trait_scores = predict_traits(input_text)
    print("\n📊 예측된 성향 점수:")
    for k, v in trait_scores.items():
        print(f"{k}: {v}")

    # 3. 성향 점수를 벡터 형태로 변환 (KNN에 입력)
    user_vec = [
        trait_scores["activity"],
        trait_scores["sociability"],
        trait_scores["care"],
        trait_scores["emotional_bond"],
        trait_scores["environment"],
        trait_scores["routine"]
    ]

    # 4. 반려동물 벡터 불러오기
    _, pet_vectors, _ = load_data_from_db()

    # 5. KNN 추천 실행
    recommended = recommend(user_vec, pet_vectors, top_k=3)

    print("\n🐾 추천된 반려동물 Top 3:")
    for i, pet_id in enumerate(recommended, start=1):
        print(f"{i}위: {pet_id}")

if __name__ == "__main__":
    main()