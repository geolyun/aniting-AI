# Top-1 Accuracy	추천 결과 중 1위가 정답 Top3에 포함되는 비율	            60% 이상
# Top-3 Accuracy	추천 결과 3개 중 정답 Top3와 얼마나 겹치는지 (Precision@3)	85% 이상
# MRR            	정답 중 가장 빠르게 맞춘 순위의 역수 평균	                0.7 이상

import pymysql
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

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


# 사용자 점수, 펫 특성, 추천 결과, 카테고리 정보 로딩
def load_data_from_db():
    conn = pymysql.connect(**db_config)
    with conn.cursor() as cur:

        # 사용자 점수 로딩 (6개 카테고리)
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

        # 펫 성향 점수 로딩 (trait_scores)
        cur.execute("SELECT pet_id, trait_scores FROM pet")
        pets = {row['pet_id']: list(map(int, row['trait_scores'].split(','))) for row in cur.fetchall()}

        # 펫 카테고리 정보 로딩
        cur.execute("SELECT pet_id, category_ids FROM pet")
        pet_category_map = {
            row['pet_id']: list(map(int, row['category_ids'].split(',')))
            for row in cur.fetchall()
        }

        # 정답 추천 이력 (Top3)
        cur.execute("SELECT users_id, top1_pet_id, top2_pet_id, top3_pet_id FROM recommend_history")
        ground_truth = {
            row['users_id']: [row['top1_pet_id'], row['top2_pet_id'], row['top3_pet_id']]
            for row in cur.fetchall()
        }

    conn.close()
    return user_vectors, pets, ground_truth, pet_category_map


# KNN 기반 추천 수행
def recommend_knn(user_vec, pet_vectors, pet_category_map, top_k=3):
    user_vec = np.array(user_vec).reshape(1, -1)

    # 상위 카테고리 2개 추출 → 관련 펫만 필터링
    top_cats = np.argsort(user_vec[0])[::-1][:2]
    preferred_cat_ids = set(top_cats + 1)  # category_id는 1부터 시작

    candidate_pets = [
        pid for pid, cats in pet_category_map.items()
        if preferred_cat_ids & set(cats)
    ]
    if not candidate_pets:
        candidate_pets = list(pet_vectors.keys())

    # 후보 펫 벡터 구성
    candidate_vectors = [pet_vectors[pid] for pid in candidate_pets]

    # KNN 모델 학습 및 최근접 이웃 검색
    knn = NearestNeighbors(n_neighbors=top_k, metric='euclidean')
    knn.fit(candidate_vectors)
    dists, indices = knn.kneighbors(user_vec)

    # 예측된 펫 ID 반환
    return [candidate_pets[i] for i in indices[0]]


# 평가 지표: Precision@3
def precision_at_3(pred, actual):
    return len(set(pred) & set(actual)) / 3

# 평가 지표: MRR (Reciprocal Rank)
def reciprocal_rank(pred, actual):
    for i, p in enumerate(pred):
        if p in actual:
            return 1 / (i + 1)
    return 0

# 평가 지표: Top-1 정답 포함 여부
def top1_hit(pred, actual):
    return int(pred[0] in actual)


# 전체 실행 흐름
def main():
    user_vectors, pet_vectors, ground_truth, pet_category_map = load_data_from_db()

    total_p3 = 0
    total_rr = 0
    total_top1 = 0
    count = 0

    print("\n[추천 결과 및 평가]")

    for uid, user_vec in tqdm(user_vectors.items()):
        if uid not in ground_truth:
            continue

        predicted = recommend_knn(user_vec, pet_vectors, pet_category_map)
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