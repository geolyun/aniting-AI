import pymysql
import pandas as pd
import torch
import os
from dotenv import load_dotenv
from transformers import BertTokenizer, BertModel
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from tqdm import tqdm

load_dotenv()

# DB 연결 정보
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'charset': os.getenv('DB_CHARSET')
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
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()

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

    # XGBoost 모델 학습
    xgb = MultiOutputRegressor(XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ))
    xgb.fit(X_train, y_train)

    # 평가
    y_pred = xgb.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    for i, trait in enumerate(expected_traits):
        print(f"{trait} MSE: {mse[i]:.4f}")

    # 모델 저장
    joblib.dump(xgb, "xgb_model.pkl")
    print("✅ XGBoost 모델이 xgb_model.pkl 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()
