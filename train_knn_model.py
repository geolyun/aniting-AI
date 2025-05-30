# Top-1 Accuracy	추천 결과 중 1위가 정답 Top3에 포함되는 비율	            60% 이상
# Top-3 Accuracy	추천 결과 3개 중 정답 Top3와 얼마나 겹치는지 (Precision@3)	85% 이상
# MRR            	정답 중 가장 빠르게 맞춘 순위의 역수 평균	                0.7 이상

import pymysql
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm
from collections import Counter

# 환경변수로 DB 접속 정보 로딩
load_dotenv()

# DB 설정
# db_config = {
#     'host': 'localhost',
#     'user': 'root',
#     'password': '1234',
#     'database': 'ARMS',
#     'charset': 'utf8mb4',
#     'cursorclass': pymysql.cursors.DictCursor
# }
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'charset': os.getenv('DB_CHARSET'),
    'cursorclass': pymysql.cursors.DictCursor
}

# DB에서 사용자 점수, 펫 특성, 정답 추천 결과, 펫 카테고리 정보를 로드
def load_data_from_db():
    conn = pymysql.connect(**db_config)
    with conn.cursor() as cur:
        # 사용자 벡터 생성
        cur.execute("""
        SELECT users_id, category_id, score_value FROM score
        WHERE users_id LIKE 'gpt_user_%'
        """)
        user_vectors = {}
        for row in cur.fetchall():
            uid = row['users_id']
            cid = row['category_id'] - 1
            score = int(row['score_value'])
            if uid not in user_vectors:
                user_vectors[uid] = [0] * 6
            user_vectors[uid][cid] = score

        # 펫 벡터 로딩
        cur.execute("SELECT pet_id, trait_scores FROM pet")
        pets = {row['pet_id']: list(map(int, row['trait_scores'].split(','))) for row in cur.fetchall()}

        # 펫 카테고리 정보 로딩
        cur.execute("SELECT pet_id, category_ids FROM pet")
        pet_category_map = {}
        for row in cur.fetchall():
            pet_category_map[row['pet_id']] = list(map(int, row['category_ids'].split(',')))

        # 추천 기록 (정답 데이터) 로딩
        cur.execute("SELECT users_id, top1_pet_id, top2_pet_id, top3_pet_id FROM recommend_history")
        ground_truth = {row['users_id']: [row['top1_pet_id'], row['top2_pet_id'], row['top3_pet_id']] for row in cur.fetchall()}

    conn.close()
    return user_vectors, pets, ground_truth, pet_category_map

# 추천 함수: 사용자 성향 벡터 → 추천 반려동물 ID 3개 반환
def recommend(user_vec, pet_vectors, pet_category_map, top_k=3):
    user_vec = normalize(np.array(user_vec).reshape(1, -1))
    
    # 성향 점수 기준 상위 3개 카테고리 추출
    top_cats = np.argsort(user_vec[0])[::-1][:3]
    weights = np.linspace(1.0, 0.5, num=3)

    # 유사 카테고리를 가진 반려동물만 필터링
    candidate_pets = []
    for pid, cats in pet_category_map.items():
        match_score = sum(weights[i] for i, c in enumerate(top_cats) if (c + 1) in cats)
        if match_score > 0.7:
            candidate_pets.append(pid)
    if not candidate_pets:
        candidate_pets = list(pet_vectors.keys())

    # 하이브리드 유사도 계산: 코사인 + 유클리디안 거리 기반
    pet_matrix = np.array([pet_vectors[pid] for pid in candidate_pets])
    cos_sim = cosine_similarity(user_vec, pet_matrix)[0]
    euclidean_dist = np.linalg.norm(user_vec - pet_matrix, axis=1)
    euclidean_sim = 1 / (1 + euclidean_dist)
    hybrid_sim = 0.7 * cos_sim + 0.3 * euclidean_sim

    # 추천 다양성 향상을 위한 보너스 점수 부여
    unique_categories = list(set(c for pid in candidate_pets for c in pet_category_map[pid]))
    diversity_bonus = [len(set(pet_category_map[pid]) & set(unique_categories)) for pid in candidate_pets]
    final_scores = hybrid_sim + 0.2 * np.array(diversity_bonus)

    # 상위 top_k개의 반려동물 ID 반환
    top_indices = np.argsort(final_scores)[::-1][:top_k * 2]
    return [candidate_pets[i] for i in top_indices[:top_k]]

# 평가 함수들
def precision_at_3(pred, actual):
    return len(set(pred) & set(actual)) / 3

def reciprocal_rank(pred, actual):
    for i, p in enumerate(pred):
        if p in actual:
            return 1 / (i + 1)
    return 0

def top1_hit(pred, actual):
    return int(pred[0] in actual)

def ndcg_at_3(pred, actual):
    dcg = sum(1 / np.log2(i + 2) for i, p in enumerate(pred[:3]) if p in actual)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(actual), 3)))
    return dcg / idcg if idcg > 0 else 0

# 전체 실행 흐름
def main():
    user_vectors, pet_vectors, ground_truth, pet_category_map = load_data_from_db()

    total_p3 = total_rr = total_top1 = total_ndcg = count = 0
    print("\n[추천 결과 및 평가]")

    for uid, user_vec in tqdm(user_vectors.items()):
        if uid not in ground_truth:
            continue

        pred = recommend(user_vec, pet_vectors, pet_category_map, top_k=3)
        actual = ground_truth[uid]

        p3 = precision_at_3(pred, actual)
        rr = reciprocal_rank(pred, actual)
        t1 = top1_hit(pred, actual)
        ndcg = ndcg_at_3(pred, actual)

        tqdm.write(f"{uid} → 예측: {pred}, 정답: {actual}, P@3: {p3:.2f}, RR: {rr:.2f}, NDCG@3: {ndcg:.2f}")

        total_p3 += p3
        total_rr += rr
        total_top1 += t1
        total_ndcg += ndcg
        count += 1

    if count:
        print("\n평가 요약")
        print(f"- 평균 Precision@3 (Top-3 Accuracy): {total_p3 / count:.2f}")
        print(f"- Top-1 Accuracy: {total_top1 / count:.2f}")
        print(f"- 평균 MRR: {total_rr / count:.2f}")
        print(f"- 평균 NDCG@3: {total_ndcg / count:.2f}")
    else:
        print("평가할 사용자 데이터가 없습니다.")

if __name__ == "__main__":
    main()