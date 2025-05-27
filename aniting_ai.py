import torch
import joblib
import numpy as np
import pymysql
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import euclidean_distances

# ========== 1. 초기화 ==========
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
bert_model = BertModel.from_pretrained("monologg/kobert")
bert_model.eval()
model = joblib.load("rf_model.pkl")

expected_traits = ["activity", "sociability", "care", "emotional_bond", "environment", "routine"]

db_config = {
    'host': 'aniting-db.cnew8oieks1a.ap-northeast-2.rds.amazonaws.com',
    'user': 'admin',
    'password': 'admin123',
    'database': 'aniting',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# ========== 2. 임베딩 함수 ==========
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.pooler_output.squeeze(0).numpy().reshape(1, -1)

# ========== 3. 성향 예측 ==========
def predict_traits(input_text: str) -> dict:
    emb = get_embedding(input_text)
    prediction = model.predict(emb)[0]
    return {
        trait: min(5, max(1, int(round(score))))
        for trait, score in zip(expected_traits, prediction)
    }

# ========== 4. 펫 정보 로딩 ==========
def load_pet_vectors():
    conn = pymysql.connect(**db_config)
    with conn.cursor() as cur:
        cur.execute("SELECT pet_id, TRAIT_scoreS FROM pet")
        pets = {row['pet_id']: list(map(int, row['TRAIT_scoreS'].split(','))) for row in cur.fetchall()}
    conn.close()
    return pets

# ========== 5. KNN 추천 ==========
def recommend_pets(user_vec: list, top_k=3):
    pet_vectors = load_pet_vectors()
    pet_ids = list(pet_vectors.keys())
    pet_matrix = np.array([pet_vectors[pid] for pid in pet_ids])
    user_matrix = np.array(user_vec).reshape(1, -1)
    distances = euclidean_distances(user_matrix, pet_matrix)[0]
    top_indices = np.argsort(distances)[:top_k]
    return [pet_ids[i] for i in top_indices]

# ========== 6. 전체 추천 파이프라인 ==========
def get_pet_recommendations(text: str) -> dict:
    traits = predict_traits(text)
    trait_vec = [traits[trait] for trait in expected_traits]
    pet_ids = recommend_pets(trait_vec)
    return {
        "traits": traits,
        "recommendations": pet_ids
    }