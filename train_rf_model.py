import pymysql
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from tqdm import tqdm

# DB 연결 정보
db_config = {
    'host': 'aniting-db.cnew8oieks1a.ap-northeast-2.rds.amazonaws.com',
    'user': 'aniting_user',
    'password': 'aniting123',
    'database': 'aniting',
    'charset': 'utf8mb4'
}

# 데이터 로드
def load_data_from_db():
    conn = pymysql.connect(**db_config)
    query = """
    SELECT r.users_id, r.question, r.category, r.answer,
           s.score_value, c.category AS trait
    FROM recommend_response r
    JOIN score s ON r.users_id = s.users_id
    JOIN category c ON s.category_id = c.category_id
    WHERE r.users_id LIKE 'gpt_user_%'
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# KoBERT 임베딩 함수
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
bert_model = BertModel.from_pretrained("monologg/kobert")
bert_model.eval()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.pooler_output.squeeze(0).numpy()

# 메인 로직
def main():
    df = load_data_from_db()

    # 입력 텍스트 구성
    df['input_text'] = df['category'] + ": " + df['question'] + " [SEP] " + df['answer']
    input_texts = df.drop_duplicates(subset='users_id')[['users_id', 'input_text']]

    # 성향 점수 피벗 처리
    score_df = df.pivot_table(index='users_id', columns='trait', values='score_value').reset_index()

    # 병합
    merged = pd.merge(input_texts, score_df, on='users_id')

    # 누락된 trait 보완
    expected_traits = ["activity", "sociability", "care", "emotional_bond", "environment", "routine"]
    for trait in expected_traits:
        if trait not in merged.columns:
            merged[trait] = 3  # 기본값 설정

    # 임베딩 및 라벨 추출
    features, labels = [], []
    for _, row in tqdm(merged.iterrows(), total=len(merged)):
        emb = get_embedding(row['input_text'])
        features.append(emb)
        labels.append([row[trait] for trait in expected_traits])

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 모델 학습
    rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    rf.fit(X_train, y_train)

    # 평가
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    for i, trait in enumerate(expected_traits):
        print(f"{trait} MSE: {mse[i]:.4f}")

    # 저장
    joblib.dump(rf, "rf_model.pkl")
    print("✅ 모델이 rf_model.pkl 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()