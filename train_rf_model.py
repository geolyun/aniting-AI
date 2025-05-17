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
    'host': 'your-db-host',
    'user': 'your-db-user',
    'password': 'your-db-password',
    'database': 'your-db-name',
    'charset': 'utf8mb4'
}

# MySQL에서 데이터 불러오기
def load_data_from_db():
    conn = pymysql.connect(**db_config)
    query = """
    SELECT r.question, r.category, r.answer, s.activity, s.sociability, s.care, s.emotional_bond, s.environment, s.routine
    FROM recommend_response r
    JOIN score s ON r.users_id = s.users_id AND r.category = s.category_code
    WHERE r.users_id LIKE 'gpt_user_%'
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# KoBERT 모델 및 토크나이저 준비
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
bert_model = BertModel.from_pretrained("monologg/kobert")
bert_model.eval()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.pooler_output.squeeze(0).numpy()

def main():
    # 1. 데이터 불러오기
    df = load_data_from_db()
    
    # 2. 텍스트 결합
    df['input_text'] = df['category'] + ": " + df['question'] + " [SEP] " + df['answer']
    
    # 3. 임베딩 및 라벨 준비
    features = []
    labels = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        emb = get_embedding(row['input_text'])
        features.append(emb)
        labels.append([
            row['activity'], row['sociability'], row['care'], 
            row['emotional_bond'], row['environment'], row['routine']
        ])
    
    # 4. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # 5. 모델 학습
    rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    rf.fit(X_train, y_train)
    
    # 6. 평가
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    traits = ["activity", "sociability", "care", "emotional_bond", "environment", "routine"]
    for i, trait in enumerate(traits):
        print(f"{trait} MSE: {mse[i]:.4f}")
    
    # 7. 모델 저장
    joblib.dump(rf, "rf_model.pkl")
    print("모델이 rf_model.pkl 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()
