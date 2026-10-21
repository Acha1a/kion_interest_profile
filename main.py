from zipfile import ZipFile
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, diags

with ZipFile("interactions.csv.zip") as z:
    with z.open("interactions.csv") as f:
        interactions = pd.read_csv(f)

# Преобразуем даты; некорректные значения станут NaT
interactions["last_watch_dt"] = pd.to_datetime(
    interactions["last_watch_dt"],
    errors="coerce"
)

bad_dates = interactions["last_watch_dt"].isna().sum()
interactions = interactions.dropna(subset=["last_watch_dt"]).copy()

# Глобальный временной разрез: ранние события — обучение,
# более поздние — проверка
cutoff = interactions["last_watch_dt"].quantile(0.8)
train = interactions[interactions["last_watch_dt"] < cutoff]
test = interactions[interactions["last_watch_dt"] >= cutoff]

train_counts = train.groupby("user_id").size()
test_user_ids = test["user_id"].unique()

# Для оценки нужны пользователи с историей и событиями в тестовом периоде
eligible_users = train_counts[train_counts >= 2].index.intersection(test_user_ids)

print("Исключено строк с некорректной датой:", bad_dates)
print("Дата разделения:", cutoff)
print("Событий в обучении:", len(train))
print("Событий в тесте:", len(test))
print("Пользователей для оценки:", len(eligible_users))
print("Период обучения:", train["last_watch_dt"].min(), "—", train["last_watch_dt"].max())
print("Период теста:", test["last_watch_dt"].min(), "—", test["last_watch_dt"].max())

with ZipFile("data_original.zip") as z:
    print([name for name in z.namelist() if name.endswith("items.csv")])

    items_file = next(
        name for name in z.namelist()
        if name.endswith("items.csv")
    )

    with z.open(items_file) as f:
        items = pd.read_csv(f)

print(items.shape)
print(items.columns.tolist())
print(items["genres"].head(10).to_list())

interaction_item_ids = set(interactions["item_id"].unique())
metadata_item_ids = set(items["item_id"].unique())

print("Уникальных объектов во взаимодействиях:", len(interaction_item_ids))
print("Объектов с метаданными:", len(metadata_item_ids))
print("Объектов без метаданных:", len(interaction_item_ids - metadata_item_ids))

# Подготовим список жанровых тегов для каждого объекта
item_genres = items[["item_id", "genres"]].copy()
item_genres["genre_list"] = (
    item_genres["genres"]
    .str.split(r"\s*,\s*")
)

# Добавим к обучающим взаимодействиям жанры объектов
train_with_genres = train[["user_id", "item_id"]].merge(
    item_genres[["item_id", "genre_list"]],
    on="item_id",
    how="left",
    validate="many_to_one"
)

# Развернём список: одна строка на тег пользователя
user_genres = (
    train_with_genres[["user_id", "genre_list"]]
    .explode("genre_list")
    .dropna(subset=["genre_list"])
)

# Считаем, сколько раз каждый пользователь взаимодействовал
# с объектами, помеченными каждым тегом
user_genre_profile = (
    user_genres
    .groupby(["user_id", "genre_list"])
    .size()
    .rename("count")
    .reset_index()
)

# Нормируем веса: у каждого пользователя они суммируются до 1
user_genre_profile["weight"] = (
    user_genre_profile["count"]
    / user_genre_profile.groupby("user_id")["count"].transform("sum")
)

print("Строк в профилях:", len(user_genre_profile))
print("Профилей пользователей:", user_genre_profile["user_id"].nunique())

print(
    user_genre_profile[
        user_genre_profile["user_id"] == eligible_users[0]
    ].sort_values("weight", ascending=False).head(10)
)

# Оцениваем топ-10 рекомендаций
K = 10

# В кандидаты включаем только объекты, встречавшиеся в train
candidate_item_ids = sorted(train["item_id"].unique())
candidate_item_set = set(candidate_item_ids)

# Метаданные кандидатов; каждый item_id должен встретиться один раз
item_meta = (
    items[items["item_id"].isin(candidate_item_set)]
    .drop_duplicates("item_id")
    .set_index("item_id")
    .reindex(candidate_item_ids)
)

# genres — строка с тегами через запятую
item_genre_lists = [
    sorted({
        genre.strip()
        for genre in str(value).split(",")
        if genre.strip()
    })
    for value in item_meta["genres"]
]

# Кодируем теги числами и строим разреженную матрицу:
# строки — объекты, столбцы — жанровые теги
all_genres = sorted({
    genre
    for genre_list in item_genre_lists
    for genre in genre_list
})
genre_to_idx = {genre: idx for idx, genre in enumerate(all_genres)}
item_to_idx = {item_id: idx for idx, item_id in enumerate(candidate_item_ids)}

genre_rows, genre_cols = [], []
for item_idx, genre_list in enumerate(item_genre_lists):
    for genre in genre_list:
        genre_rows.append(item_idx)
        genre_cols.append(genre_to_idx[genre])

item_genre_matrix = csr_matrix(
    (
        np.ones(len(genre_rows), dtype=np.float32),
        (genre_rows, genre_cols),
    ),
    shape=(len(candidate_item_ids), len(all_genres)),
)

# Строим матрицу пользователь–объект только для пользователей оценки
eval_user_ids = list(eligible_users)
user_to_idx = {user_id: idx for idx, user_id in enumerate(eval_user_ids)}

train_eval = train[train["user_id"].isin(eval_user_ids)].copy()
train_eval["u_idx"] = train_eval["user_id"].map(user_to_idx)
train_eval["i_idx"] = train_eval["item_id"].map(item_to_idx)

user_item_matrix = csr_matrix(
    (
        np.ones(len(train_eval), dtype=np.float32),
        (train_eval["u_idx"], train_eval["i_idx"]),
    ),
    shape=(len(eval_user_ids), len(candidate_item_ids)),
)

# Профиль пользователя = сумма жанровых тегов его объектов из train
user_genre_matrix = user_item_matrix @ item_genre_matrix

# Нормируем веса каждого пользователя до суммы 1
row_sums = np.asarray(user_genre_matrix.sum(axis=1)).ravel()
inverse_sums = np.zeros_like(row_sums, dtype=np.float32)
np.divide(1.0, row_sums, out=inverse_sums, where=row_sums != 0)

user_profile_matrix = diags(inverse_sums) @ user_genre_matrix

# Считаем оценки объектов порциями, чтобы не создавать
# огромную матрицу оценок целиком
recommendations = {}
batch_size = 256

for start in range(0, len(eval_user_ids), batch_size):
    end = min(start + batch_size, len(eval_user_ids))

    scores_batch = (
        user_profile_matrix[start:end] @ item_genre_matrix.T
    ).toarray()

    for local_idx, user_idx in enumerate(range(start, end)):
        # Исключаем объекты, которые пользователь уже видел в train
        seen_start = user_item_matrix.indptr[user_idx]
        seen_end = user_item_matrix.indptr[user_idx + 1]
        seen_item_indices = user_item_matrix.indices[seen_start:seen_end]
        scores_batch[local_idx, seen_item_indices] = -np.inf

        row_scores = scores_batch[local_idx]
        top_count = min(K, np.isfinite(row_scores).sum())

        if top_count == 0:
            recommendations[eval_user_ids[user_idx]] = []
            continue

        top_indices = np.argpartition(-row_scores, top_count - 1)[:top_count]
        top_indices = top_indices[
            np.argsort(-row_scores[top_indices], kind="stable")
        ]

        recommendations[eval_user_ids[user_idx]] = [
            candidate_item_ids[item_idx] for item_idx in top_indices
        ]

# Релевантные объекты: взаимодействия пользователя в тестовом периоде.
# Учитываем здесь только объекты из выбранного кандидатного каталога.
test_for_eval = test[test["user_id"].isin(eval_user_ids)].copy()
test_warm = test_for_eval[
    test_for_eval["item_id"].isin(candidate_item_set)
]

relevant_by_user = (
    test_warm.groupby("user_id")["item_id"]
    .apply(set)
    .to_dict()
)

# Пользователи без тестовых объектов в кандидатном каталоге
# не входят в условный расчёт ранжирования.
# noinspection LanguageDetectionInspection
metric_users = [
    user_id for user_id in eval_user_ids
    if relevant_by_user.get(user_id)
]

recall_values = []
ndcg_values = []

for user_id in metric_users:
    relevant = relevant_by_user[user_id]
    ranked_items = recommendations.get(user_id, [])[:K]

    hits = sum(item_id in relevant for item_id in ranked_items)
    recall_values.append(hits / len(relevant))

    dcg = sum(
        1 / np.log2(rank + 2)
        for rank, item_id in enumerate(ranked_items)
        if item_id in relevant
    )

    ideal_hits = min(len(relevant), K)
    idcg = sum(
        1 / np.log2(rank + 2)
        for rank in range(ideal_hits)
    )

    ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)

test_coverage = (
    len(test_warm) / len(test_for_eval)
    if len(test_for_eval) > 0
    else 0.0
)

print("Кандидатов:", len(candidate_item_ids))
print("Пользователей в оценке:", len(metric_users))
print("Доля тестовых событий с объектами-кандидатами:", round(test_coverage, 4))
print(f"Recall@{K}: {np.mean(recall_values):.4f}")
print(f"NDCG@{K}: {np.mean(ndcg_values):.4f}")

def evaluate_recommendations(recommendations_to_eval):
    recalls = []
    ndcgs = []

    for user_id in metric_users:
        relevant = relevant_by_user[user_id]
        ranked_items = recommendations_to_eval.get(user_id, [])[:K]

        hits = sum(item_id in relevant for item_id in ranked_items)
        recalls.append(hits / len(relevant))

        dcg = sum(
            1 / np.log2(rank + 2)
            for rank, item_id in enumerate(ranked_items)
            if item_id in relevant
        )

        ideal_hits = min(len(relevant), K)
        idcg = sum(
            1 / np.log2(rank + 2)
            for rank in range(ideal_hits)
        )

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return np.mean(recalls), np.mean(ndcgs)


# Популярность объекта считаем только по обучающим данным
item_popularity = train["item_id"].value_counts().to_dict()

# Сортируем индексы объектов по популярности
popularity_order_indices = sorted(
    range(len(candidate_item_ids)),
    key=lambda idx: (
        -int(item_popularity.get(candidate_item_ids[idx], 0)),
        idx
    )
)

popularity_recommendations = {}

for user_idx, user_id in enumerate(eval_user_ids):
    start = user_item_matrix.indptr[user_idx]
    end = user_item_matrix.indptr[user_idx + 1]

    # Храним индексы уже просмотренных объектов
    seen_indices = {
        int(idx)
        for idx in user_item_matrix.indices[start:end]
    }

    popularity_recommendations[user_id] = [
        candidate_item_ids[idx]
        for idx in popularity_order_indices
        if idx not in seen_indices
    ][:K]


profile_recall, profile_ndcg = evaluate_recommendations(recommendations)
pop_recall, pop_ndcg = evaluate_recommendations(popularity_recommendations)

print(f"Жанровый профиль — Recall@{K}: {profile_recall:.4f}, NDCG@{K}: {profile_ndcg:.4f}")
print(f"Популярность     — Recall@{K}: {pop_recall:.4f}, NDCG@{K}: {pop_ndcg:.4f}")
