# Top-1 Accuracy	추천 결과 중 1위가 정답 Top3에 포함되는 비율	            60% 이상
# Top-3 Accuracy	추천 결과 3개 중 정답 Top3와 얼마나 겹치는지 (Precision@3)	85% 이상
# MRR            	정답 중 가장 빠르게 맞춘 순위의 역수 평균	                0.7 이상

import pymysql
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import precision_score
from tqdm import tqdm

# DB 설정
db_config = {
    'host': 'localhost',         # 운영 시: 'aniting-db.cnew8oieks1a.ap-northeast-2.rds.amazonaws.com'
    'user': 'root',              # 운영 시: 'admin'
    'password': '1234',          # 운영 시: 'admin123'
    'database': 'ARMS',          # 운영 시: 'aniting'
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# 사용자 점수와 추천 결과를 DB에서 불러옴
def load_data_from_db():
    conn = pymysql.connect(**db_config)
    with conn.cursor() as cur:
        # 사용자 벡터 구성
        cur.execute("""
        SELECT USERS_ID, CATEGORY_ID, SCORE_VALUE FROM SCORE
        WHERE USERS_ID LIKE 'gpt_user_%'
        """)
        user_rows = cur.fetchall()

        user_vectors = {}
        for row in user_rows:
            uid = row['USERS_ID']
            cid = row['CATEGORY_ID'] - 1  # 0-indexed
            score = int(row['SCORE_VALUE'])
            if uid not in user_vectors:
                user_vectors[uid] = [0] * 6
            user_vectors[uid][cid] = score

        # 반려동물 벡터 구성
        cur.execute("SELECT PET_ID, TRAIT_SCORES FROM PET")
        pets = {}
        for row in cur.fetchall():
            pets[row['PET_ID']] = list(map(int, row['TRAIT_SCORES'].split(',')))

        # 정답 데이터 구성
        cur.execute("SELECT USERS_ID, TOP1_PET_ID, TOP2_PET_ID, TOP3_PET_ID FROM RECOMMEND_HISTORY")
        ground_truth = {row['USERS_ID']: [row['TOP1_PET_ID'], row['TOP2_PET_ID'], row['TOP3_PET_ID']] for row in cur.fetchall()}

    conn.close()
    return user_vectors, pets, ground_truth

# KNN 추천 함수
def recommend(user_vec, pet_vectors, top_k=3):
    pet_ids = list(pet_vectors.keys())
    pet_matrix = np.array([pet_vectors[pid] for pid in pet_ids])
    user_matrix = np.array(user_vec).reshape(1, -1)

    distances = euclidean_distances(user_matrix, pet_matrix)[0]
    top_indices = np.argsort(distances)[:top_k]

    return [pet_ids[i] for i in top_indices]

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
    user_vectors, pet_vectors, ground_truth = load_data_from_db()

    total_p3 = 0
    total_rr = 0
    total_top1 = 0
    count = 0

    print("\n[추천 결과 및 평가]")

    for uid, user_vec in tqdm(user_vectors.items()):
        if uid not in ground_truth:
            continue

        predicted = recommend(user_vec, pet_vectors)
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