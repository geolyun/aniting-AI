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

# 사용자 점수, 펫 특성, 정답, 인기도, 카테고리 정보 로딩
def load_data_from_db():
    conn = pymysql.connect(**db_config)
    with conn.cursor() as cur:
        
        # 사용자 벡터 생성
        cur.execute("""
        SELECT users_id, category_id, score_value FROM score
        WHERE users_id LIKE 'gpt_user_%'
        """)
        user_rows = cur.fetchall()
        user_vectors = {}
        for row in user_rows:
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

        # 정답 Top3 추천 결과
        cur.execute("SELECT users_id, top1_pet_id, top2_pet_id, top3_pet_id FROM recommend_history")
        ground_truth = {row['users_id']: [row['top1_pet_id'], row['top2_pet_id'], row['top3_pet_id']] for row in cur.fetchall()}

        # 펫 인기 점수 계산 (Top3 빈도 기반)
        all_top3 = [pet for trio in ground_truth.values() for pet in trio]
        pet_popularity = Counter(all_top3)

    conn.close()
    return user_vectors, pets, ground_truth, pet_popularity, pet_category_map

# 추천 함수 (KNN + 인기도 점수 보정 + 카테고리 필터링 + 동적 alpha)
def recommend(user_vec, pet_vectors, pet_popularity, pet_category_map, top_k=3):
    user_vec = normalize(np.array(user_vec).reshape(1, -1))

    # 상위 2개 카테고리 추출
    top_cats = np.argsort(user_vec[0])[::-1][:2]
    preferred_cat_ids = set(top_cats + 1)  # category_id는 1부터 시작

    # 필터링된 후보 펫 추출
    candidate_pets = [
        pid for pid, cats in pet_category_map.items()
        if preferred_cat_ids & set(cats)
    ]
    if not candidate_pets:
        candidate_pets = list(pet_vectors.keys())

    # 유사도 계산
    pet_matrix = normalize(np.array([pet_vectors[pid] for pid in candidate_pets]))
    sim_scores = cosine_similarity(user_vec, pet_matrix)[0]

    # 펫 인기도 점수 계산
    max_pop = max(pet_popularity.values(), default=1)
    pop_scores = np.array([pet_popularity.get(pid, 0) / max_pop for pid in candidate_pets])

    # 사용자 벡터 분산 기반으로 동적 alpha 결정
    alpha = 0.9 if np.std(user_vec) >= 0.15 else 0.6
    hybrid_scores = alpha * sim_scores + (1 - alpha) * pop_scores

    # 상위 추천 결과 Top-K 추출
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    return [candidate_pets[i] for i in top_indices]

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

# 전체 실행 흐름
def main():
    user_vectors, pet_vectors, ground_truth, pet_popularity, pet_category_map = load_data_from_db()

    total_p3 = 0
    total_rr = 0
    total_top1 = 0
    count = 0

    print("\n[추천 결과 및 평가]")

    for uid, user_vec in tqdm(user_vectors.items()):
        if uid not in ground_truth:
            continue

        predicted = recommend(user_vec, pet_vectors, pet_popularity, pet_category_map)
        actual = ground_truth[uid]

        p3 = precision_at_3(predicted, actual)
        rr = reciprocal_rank(predicted, actual)
        t1 = top1_hit(predicted, actual)

        tqdm.write(f"{uid} → 예측: {predicted}, 정답: {actual}, P@3: {p3:.2f}, RR: {rr:.2f}")

        total_p3 += p3
        total_rr += rr
        total_top1 += t1
        count += 1

    if count:
        print("\n 평가 요약")
        print(f"- 평균 Precision@3 (Top-3 Accuracy): {total_p3 / count:.2f}")
        print(f"- Top-1 Accuracy: {total_top1 / count:.2f}")
        print(f"- 평균 MRR: {total_rr / count:.2f}")
    else:
        print("평가할 사용자 데이터가 없습니다.")

if __name__ == "__main__":
    main()