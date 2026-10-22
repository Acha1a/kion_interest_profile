from zipfile import ZipFile
import pandas as pd
import numpy as np
import time
from scipy.sparse import csr_matrix, diags
import re

def _parse_genres(value):
    """Преобразует строку жанров или список жанров в нормализованный список."""
    if isinstance(value, (list, tuple, set)):
        values = value
    elif pd.isna(value):
        return []
    else:
        # Поддерживает разделители запятая, точка с запятой и |
        text = str(value).strip().strip("[]")
        values = re.split(r"[,;|]", text)

    return [
        str(genre).strip().strip("'\"").lower()
        for genre in values
        if str(genre).strip().strip("'\"")
    ]


def make_snapshot(history, items):
    """
    Строит признаки-снимок по прошлым взаимодействиям.

    history: взаимодействия только до момента предсказания
    items: метаданные объектов; нужны item_id, genres
    """
    required_history = {"user_id", "item_id"}
    required_items = {"item_id", "genres"}

    if not required_history.issubset(history.columns):
        raise ValueError(f"В history нужны колонки: {required_history}")
    if not required_items.issubset(items.columns):
        raise ValueError(f"В items нужны колонки: {required_items}")

    item_columns = ["item_id", "genres"]
    for col in ("content_type", "release_year"):
        if col in items.columns:
            item_columns.append(col)

    item_meta = items[item_columns].drop_duplicates("item_id").copy()
    item_meta["genre_list"] = item_meta["genres"].apply(_parse_genres)

    # История пользователя и популярность объектов
    user_activity = history.groupby("user_id").size().to_dict()
    item_popularity = history.groupby("item_id").size().to_dict()

    # Профиль интересов: доля взаимодействий пользователя по каждому жанру
    hist = history[["user_id", "item_id"]].merge(
        item_meta[["item_id", "genre_list"]],
        on="item_id",
        how="left",
    )
    hist["genre_list"] = hist["genre_list"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    genre_rows = hist[["user_id", "genre_list"]].explode("genre_list")
    genre_rows = genre_rows.dropna(subset=["genre_list"])

    if genre_rows.empty:
        user_genre_profile = {}
    else:
        counts = (
            genre_rows.groupby(["user_id", "genre_list"])
            .size()
            .rename("count")
            .reset_index()
        )
        totals = counts.groupby("user_id")["count"].transform("sum")
        counts["share"] = counts["count"] / totals

        user_genre_profile = (
            counts.groupby("user_id")
            .apply(
                lambda group: dict(
                    zip(group["genre_list"], group["share"])
                ),
                include_groups=False,
            )
            .to_dict()
        )

    content_type_map = {}
    if "content_type" in item_meta.columns:
        content_types = (
            item_meta["content_type"]
            .dropna()
            .astype(str)
            .unique()
        )
        content_type_map = {
            value: code for code, value in enumerate(sorted(content_types))
        }

    return {
        "item_meta": item_meta,
        "user_activity": user_activity,
        "item_popularity": item_popularity,
        "user_genre_profile": user_genre_profile,
        "content_type_map": content_type_map,
    }

def build_pair_features(pairs, snapshot):
    """
    Создаёт числовые признаки для таблицы пар user_id–item_id.
    Возвращает строки в том же порядке, что и pairs.
    """
    required = {"user_id", "item_id"}
    if not required.issubset(pairs.columns):
        raise ValueError(f"В pairs нужны колонки: {required}")

    result = pairs[["user_id", "item_id"]].copy().reset_index(drop=True)
    result["_row_order"] = np.arange(len(result))

    meta_cols = ["item_id", "genre_list"]
    for col in ("content_type", "release_year"):
        if col in snapshot["item_meta"].columns:
            meta_cols.append(col)

    result = result.merge(
        snapshot["item_meta"][meta_cols],
        on="item_id",
        how="left",
        sort=False,
    )

    profiles = snapshot["user_genre_profile"]
    item_popularity = snapshot["item_popularity"]
    user_activity = snapshot["user_activity"]

    result["item_popularity"] = (
        result["item_id"].map(item_popularity).fillna(0).astype(float)
    )
    result["user_activity"] = (
        result["user_id"].map(user_activity).fillna(0).astype(float)
    )

    result["item_popularity_log"] = np.log1p(result["item_popularity"])
    result["user_activity_log"] = np.log1p(result["user_activity"])

    def genre_scores(row):
        genres = row["genre_list"]
        if not isinstance(genres, list):
            genres = []

        profile = profiles.get(row["user_id"], {})
        scores = [profile.get(genre, 0.0) for genre in genres]

        if not scores:
            return pd.Series({
                "genre_interest_sum": 0.0,
                "genre_interest_max": 0.0,
                "genre_count": 0.0,
            })

        return pd.Series({
            "genre_interest_sum": float(sum(scores)),
            "genre_interest_max": float(max(scores)),
            "genre_count": float(len(genres)),
        })

    genre_features = result.apply(genre_scores, axis=1)
    result = pd.concat([result, genre_features], axis=1)

    if "release_year" in result.columns:
        result["release_year"] = pd.to_numeric(
            result["release_year"], errors="coerce"
        ).fillna(0)

    if "content_type" in result.columns:
        result["content_type_id"] = (
            result["content_type"]
            .astype(str)
            .map(snapshot["content_type_map"])
            .fillna(-1)
            .astype(int)
        )

    # Только числовые признаки — без user_id и item_id
    feature_cols = [
        "item_popularity_log",
        "user_activity_log",
        "genre_interest_sum",
        "genre_interest_max",
        "genre_count",
    ]
    for col in ("release_year", "content_type_id"):
        if col in result.columns:
            feature_cols.append(col)

    return (
        result.sort_values("_row_order")[feature_cols]
        .reset_index(drop=True)
    )

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

# Сохраняем внешнее разбиение:
# train — до августа, test — 2021-08-04 — 2021-08-22
final_train = train.copy()
final_test = test.copy()

# Последние примерно 20% исходного train используем для валидации
validation_cutoff = final_train["last_watch_dt"].quantile(0.8)

train = final_train[
    final_train["last_watch_dt"] < validation_cutoff
].copy()

test = final_train[
    final_train["last_watch_dt"] >= validation_cutoff
].copy()

# Пересчитываем пользователей для оценки на новой валидационной выборке
train_counts = train.groupby("user_id").size()
valid_user_ids = test["user_id"].unique()

eligible_users = train_counts[train_counts >= 2].index.intersection(
    valid_user_ids
)

print("\n--- Внутренняя валидация ---")
print("Дата разделения:", validation_cutoff)
print("Событий для обучения:", len(train))
print("Событий для валидации:", len(test))
print("Пользователей для оценки:", len(eligible_users))
print(
    "Период обучения:",
    train["last_watch_dt"].min(),
    "—",
    train["last_watch_dt"].max()
)
print(
    "Период валидации:",
    test["last_watch_dt"].min(),
    "—",
    test["last_watch_dt"].max()
)

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

with ZipFile("data_original.zip") as z:
    users_file = next(
        name for name in z.namelist()
        if name.replace("\\", "/").split("/")[-1] == "users.csv"
        and not name.replace("\\", "/").startswith("__MACOSX/")
    )

    with z.open(users_file) as f:
        users = pd.read_csv(f)

print("Размер users:", users.shape)
print("Колонки users:", users.columns.tolist())
print(users.head())

try:
    import sklearn
    print("Версия scikit-learn:", sklearn.__version__)
except ImportError:
    print("scikit-learn не установлен")

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

from sklearn.ensemble import HistGradientBoostingClassifier

# Сначала фиксируем статические признаки.
# Демография и метаданные используются как признаки, а не как сигналы будущих просмотров.
user_static = (
    users.drop_duplicates("user_id")
    .set_index("user_id")
)

user_cat_cols = ["age", "income", "sex", "kids_flg"]
user_cat = pd.get_dummies(
    user_static[user_cat_cols].fillna("unknown").astype(str),
    prefix=user_cat_cols,
    dtype=np.float32,
)

item_static = (
    items.drop_duplicates("item_id")
    .set_index("item_id")
)

item_cat_cols = ["content_type", "for_kids", "age_rating"]
item_cat = pd.get_dummies(
    item_static[item_cat_cols].astype("object").fillna("unknown").astype(str),
    prefix=item_cat_cols,
    dtype=np.float32,
)

release_year = pd.to_numeric(
    item_static["release_year"], errors="coerce"
).fillna(0).astype(np.float32) / 1000

item_base = item_cat.copy()
item_base["release_year_scaled"] = release_year

# Жанры — статические метаданные объектов.
item_genres_map = {}
for row in items[["item_id", "genres"]].drop_duplicates("item_id").itertuples(index=False):
    value = row.genres
    if pd.isna(value):
        item_genres_map[row.item_id] = []
    else:
        item_genres_map[row.item_id] = [
            genre.strip() for genre in str(value).split(",") if genre.strip()
        ]


def make_snapshot(history, candidate_ids):
    """Строит признаки и жанровые профили только по переданной истории."""
    user_features = user_cat.copy()
    user_counts = history.groupby("user_id").size()
    user_features["log_user_events"] = np.log1p(
        user_counts.reindex(user_features.index).fillna(0)
    ).astype(np.float32)

    item_features = item_base.reindex(candidate_ids).fillna(0).copy()
    item_counts = history["item_id"].value_counts()
    item_features["log_item_popularity"] = np.log1p(
        item_counts.reindex(item_features.index).fillna(0)
    ).astype(np.float32)

    # Жанровые интересы считаются по истории, а не по целевому периоду.
    genre_rows = history[["user_id", "item_id"]].copy()
    genre_rows["genre"] = genre_rows["item_id"].map(item_genres_map)
    genre_rows = genre_rows.explode("genre").dropna(subset=["genre"])

    genre_counts = (
        genre_rows.groupby(["user_id", "genre"])
        .size()
        .rename("count")
        .reset_index()
    )
    genre_counts["weight"] = (
        genre_counts["count"]
        / genre_counts.groupby("user_id")["count"].transform("sum")
    )

    profiles = {}
    for user_id, group in genre_counts.groupby("user_id"):
        profiles[user_id] = dict(zip(group["genre"], group["weight"]))

    return user_features, item_features, profiles


def make_pair_features(user_ids, item_ids, user_features,
                       item_features, profiles):
    """Превращает пары пользователь–объект в числовые признаки."""
    user_matrix = (
        user_features.reindex(user_ids)
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    item_matrix = (
        item_features.reindex(item_ids)
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )

    genre_match = np.fromiter(
        (
            sum(
                profiles.get(user_id, {}).get(genre, 0.0)
                for genre in item_genres_map.get(item_id, [])
            )
            for user_id, item_id in zip(user_ids, item_ids)
        ),
        dtype=np.float32,
        count=len(user_ids),
    ).reshape(-1, 1)

    return np.concatenate([user_matrix, item_matrix, genre_match], axis=1)


# Внутренний временной разрез для обучения бустинга:
# ранняя часть — история для признаков, поздняя — целевые метки.
gbm_cutoff = train["last_watch_dt"].quantile(0.8)
gbm_target_start = pd.Timestamp("2021-06-20")
gbm_history = train[
    train["last_watch_dt"] < gbm_target_start
].copy()
gbm_target = train[
    train["last_watch_dt"] >= gbm_target_start
].copy()



gbm_candidate_ids = np.array(sorted(gbm_history["item_id"].unique()))
gbm_candidate_set = set(gbm_candidate_ids)

# Положительные примеры — целевые взаимодействия с объектами,
# уже присутствовавшими в каталоге на момент построения истории.
positive_pairs_all = (
    gbm_target[["user_id", "item_id"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

seen_in_gbm_history = (
    gbm_history.groupby("user_id")["item_id"].agg(set).to_dict()
)
positive_in_gbm_target = (
    positive_pairs_all.groupby("user_id")["item_id"].agg(set).to_dict()
)
gbm_candidate_ids = np.asarray(candidate_item_ids)


# train — внутреннее обучение до 2021-07-17.
# test — внутренняя валидация: 2021-07-17 — 2021-08-03.
train["last_watch_dt"] = pd.to_datetime(train["last_watch_dt"])
test["last_watch_dt"] = pd.to_datetime(test["last_watch_dt"])

assert train["last_watch_dt"].max() < pd.Timestamp("2021-07-17")
assert test["last_watch_dt"].min() >= pd.Timestamp("2021-07-17")
assert test["last_watch_dt"].max() <= pd.Timestamp("2021-08-03")

# Дополнительная временная отсечка внутри train:
# историю используем для признаков, более поздние события — как обучающие метки.
gbm_target_start = pd.Timestamp("2021-06-20")

gbm_history = final_train[
    final_train["last_watch_dt"] < validation_cutoff
].copy()


gbm_target = final_train[
    final_train["last_watch_dt"] >= validation_cutoff
].copy()

positive_pairs_all = (
    gbm_target[["user_id", "item_id"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

seen_in_gbm_history = (
    gbm_history.groupby("user_id")["item_id"].agg(set).to_dict()
)
positive_in_gbm_target = (
    gbm_target.groupby("user_id")["item_id"].agg(set).to_dict()
)

gbm_candidate_ids = np.asarray(candidate_item_ids)

# Ограничиваем число обучающих позитивов, чтобы прототип укладывался
# в доступное время и память.
MAX_POSITIVES = 80_000
NEGATIVES_PER_POSITIVE = 4
rng = np.random.default_rng(42)

if len(positive_pairs_all) > MAX_POSITIVES:
    positive_pairs = positive_pairs_all.sample(
        n=MAX_POSITIVES, random_state=42
    ).reset_index(drop=True)
else:
    positive_pairs = positive_pairs_all.reset_index(drop=True)

negative_rows = []

for user_id in positive_pairs["user_id"].to_numpy():
    forbidden = (
        seen_in_gbm_history.get(user_id, set())
        | positive_in_gbm_target.get(user_id, set())
    )

    selected = set()
    attempts = 0

    while len(selected) < NEGATIVES_PER_POSITIVE and attempts < 20:
        draws = rng.choice(gbm_candidate_ids, size=64, replace=True)
        for item_id in draws:
            if item_id not in forbidden:
                selected.add(item_id)
                if len(selected) == NEGATIVES_PER_POSITIVE:
                    break
        attempts += 1

    negative_rows.extend(
        (user_id, item_id) for item_id in selected
    )

negative_pairs = pd.DataFrame(
    negative_rows, columns=["user_id", "item_id"]
)

gbm_train_pairs = pd.concat(
    [
        positive_pairs.assign(label=1),
        negative_pairs.assign(label=0),
    ],
    ignore_index=True,
)

gbm_user_features, gbm_item_features, gbm_profiles = make_snapshot(
    gbm_history, gbm_candidate_ids
)

t0 = time.perf_counter()

X_gbm = make_pair_features(
    gbm_train_pairs["user_id"].to_numpy(),
    gbm_train_pairs["item_id"].to_numpy(),
    gbm_user_features,
    gbm_item_features,
    gbm_profiles,
)
y_gbm = gbm_train_pairs["label"].to_numpy(dtype=np.int8)

print(f"Подготовка обучающих пар: {time.perf_counter() - t0:.1f} сек.")
print("Обучающих пар для GBM:", len(gbm_train_pairs))
print("Положительных:", int(y_gbm.sum()))
print("Отрицательных:", int((y_gbm == 0).sum()))

gbm_model = HistGradientBoostingClassifier(
    max_iter=100,
    max_leaf_nodes=15,
    learning_rate=0.08,
    random_state=42,
)
t0 = time.perf_counter()
gbm_model.fit(X_gbm, y_gbm)
print(f"Обучение GBM: {time.perf_counter() - t0:.1f} сек.")

# Для валидации признаки строим по всему внутреннему train.
# Переменная test здесь — внутренняя валидация, final_test не используется.
valid_candidate_ids = np.array(candidate_item_ids)
valid_user_features, valid_item_features, valid_profiles = make_snapshot(
    train, valid_candidate_ids
)
seen_in_valid_history = train.groupby("user_id")["item_id"].agg(set).to_dict()

# Чтобы первая проверка модели не считала сотни миллионов пар,
# оцениваем случайную выборку из 1 000 пользователей.
gbm_eval_users = np.array(metric_users)
if len(gbm_eval_users) > 1_000:
    gbm_eval_users = rng.choice(gbm_eval_users, size=1_000, replace=False)

gbm_recommendations = {}
candidate_count = len(valid_candidate_ids)
users_per_batch = 8

for start in range(0, len(gbm_eval_users), users_per_batch):
    batch_users = gbm_eval_users[start:start + users_per_batch]

    pair_users = np.repeat(batch_users, candidate_count)
    pair_items = np.tile(valid_candidate_ids, len(batch_users))

    X_batch = make_pair_features(
        pair_users,
        pair_items,
        valid_user_features,
        valid_item_features,
        valid_profiles,
    )

    batch_scores = gbm_model.predict_proba(X_batch)[:, 1].reshape(
        len(batch_users), candidate_count
    )

    for row_idx, user_id in enumerate(batch_users):
        seen_ids = seen_in_valid_history.get(user_id, set())
        seen_indices = [
            idx for idx, item_id in enumerate(valid_candidate_ids)
            if item_id in seen_ids
        ]
        batch_scores[row_idx, seen_indices] = -np.inf

        top_n = min(K, candidate_count)
        top_indices = np.argpartition(
            -batch_scores[row_idx], top_n - 1
        )[:top_n]
        top_indices = top_indices[
            np.argsort(-batch_scores[row_idx][top_indices], kind="stable")
        ]

        gbm_recommendations[user_id] = [
            valid_candidate_ids[idx] for idx in top_indices
        ]


def metrics_for_users(recommendation_dict, users_to_score):
    t0 = time.perf_counter()
    recalls, ndcgs = [], []

    for user_id in users_to_score:
        relevant = relevant_by_user.get(user_id, set())
        if not relevant:
            continue

        ranked = recommendation_dict.get(user_id, [])[:K]
        hits = sum(item_id in relevant for item_id in ranked)
        recalls.append(hits / len(relevant))

        dcg = sum(
            1 / np.log2(rank + 2)
            for rank, item_id in enumerate(ranked)
            if item_id in relevant
        )
        ideal_hits = min(len(relevant), K)
        idcg = sum(1 / np.log2(rank + 2) for rank in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg else 0.0)

    print(
        f"Расчёт метрик для {len(users_to_score)} пользователей: "
        f"{time.perf_counter() - t0:.2f} сек."
    )
    return np.mean(recalls), np.mean(ndcgs)


print("\n--- Сравнение на одной выборке пользователей ---")
for name, recs in [
    ("Жанровый профиль", recommendations),
    ("Популярность", popularity_recommendations),
    ("Gradient Boosting", gbm_recommendations),
]:
    recall, ndcg = metrics_for_users(recs, gbm_eval_users)
    print(f"{name:20s} Recall@{K}: {recall:.4f}, NDCG@{K}: {ndcg:.4f}")

# ================= ФИНАЛЬНОЕ ОБУЧЕНИЕ И ТЕСТ =================

final_train["last_watch_dt"] = pd.to_datetime(final_train["last_watch_dt"])
final_test["last_watch_dt"] = pd.to_datetime(final_test["last_watch_dt"])

# Внутри final_train отделяем прошлую историю от более поздних
# обучающих целей. Тестовые события здесь не используются.
final_gbm_cutoff = pd.Timestamp("2021-07-17")

final_gbm_history = final_train[
    final_train["last_watch_dt"] < final_gbm_cutoff
    ].copy()

final_gbm_target = final_train[
    final_train["last_watch_dt"] >= final_gbm_cutoff
    ].copy()

# Для обучающих пар берём только объекты, уже известные по истории.
final_gbm_candidate_ids = np.asarray(
    sorted(final_gbm_history["item_id"].unique())
)
final_gbm_candidate_set = set(final_gbm_candidate_ids)

final_positive_all = (
    final_gbm_target[
        final_gbm_target["item_id"].isin(final_gbm_candidate_set)
    ][["user_id", "item_id"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

final_seen_history = (
    final_gbm_history.groupby("user_id")["item_id"].agg(set).to_dict()
)
final_positive_by_user = (
    final_gbm_target.groupby("user_id")["item_id"].agg(set).to_dict()
)

# Сэмплируем обучающие позитивы и негативы.
final_rng = np.random.default_rng(42)

if len(final_positive_all) > MAX_POSITIVES:
    final_positive = final_positive_all.sample(
        n=MAX_POSITIVES,
        random_state=42,
    ).reset_index(drop=True)
else:
    final_positive = final_positive_all.reset_index(drop=True)

final_negative_rows = []

for user_id in final_positive["user_id"].to_numpy():
    forbidden = (
            final_seen_history.get(user_id, set())
            | final_positive_by_user.get(user_id, set())
    )

    selected = set()
    attempts = 0

    while (
            len(selected) < NEGATIVES_PER_POSITIVE
            and attempts < 20
    ):
        draws = final_rng.choice(
            final_gbm_candidate_ids,
            size=64,
            replace=True,
        )

        for item_id in draws:
            if item_id not in forbidden:
                selected.add(item_id)
                if len(selected) == NEGATIVES_PER_POSITIVE:
                    break

        attempts += 1

    final_negative_rows.extend(
        (user_id, item_id) for item_id in selected
    )

final_negative = pd.DataFrame(
    final_negative_rows,
    columns=["user_id", "item_id"],
)

final_gbm_pairs = pd.concat(
    [
        final_positive.assign(label=1),
        final_negative.assign(label=0),
    ],
    ignore_index=True,
)

final_gbm_user_features, final_gbm_item_features, final_gbm_profiles = (
    make_snapshot(final_gbm_history, final_gbm_candidate_ids)
)

X_final_gbm = make_pair_features(
    final_gbm_pairs["user_id"].to_numpy(),
    final_gbm_pairs["item_id"].to_numpy(),
    final_gbm_user_features,
    final_gbm_item_features,
    final_gbm_profiles,
)
y_final_gbm = final_gbm_pairs["label"].to_numpy(dtype=np.int8)

print("\n--- Финальное обучение GBM ---")
print("Обучающих пар:", len(final_gbm_pairs))
print("Положительных:", int(y_final_gbm.sum()))
print("Отрицательных:", int((y_final_gbm == 0).sum()))

final_gbm_model = HistGradientBoostingClassifier(
    max_iter=100,
    max_leaf_nodes=15,
    learning_rate=0.08,
    random_state=42,
)
final_gbm_model.fit(X_final_gbm, y_final_gbm)

# Для августовского теста доступны все объекты из final_train.
final_candidate_ids = np.asarray(
    sorted(final_train["item_id"].unique())
)

# Здесь snapshot строится по всей истории до 2021-08-04.
final_user_features, final_item_features, final_profiles = make_snapshot(
    final_train,
    final_candidate_ids,
)

final_seen_by_user = (
    final_train.groupby("user_id")["item_id"].agg(set).to_dict()
)
final_relevant_by_user = (
    final_test.groupby("user_id")["item_id"].agg(set).to_dict()
)

# Пользователь должен иметь историю в обучении и события в тесте.
final_eligible_users = (
        set(final_train["user_id"].unique())
        & set(final_test["user_id"].unique())
)

# Оставляем пользователей из заранее подготовленного финального списка,
# у которых есть тестовые события.
final_eval_users = np.asarray([
    user_id
    for user_id in final_eligible_users
    if user_id in final_relevant_by_user
])

# Быстрый прогон на 1 000 пользователях. Для оценки всех пользователей
# замени 1_000 на len(final_eval_users).
FINAL_EVAL_SAMPLE = min(1_000, len(final_eval_users))
final_eval_rng = np.random.default_rng(42)

if len(final_eval_users) > FINAL_EVAL_SAMPLE:
    final_eval_users = final_eval_rng.choice(
        final_eval_users,
        size=FINAL_EVAL_SAMPLE,
        replace=False,
    )

# Ранжируем кандидатов GBM батчами.
final_gbm_recommendations = {}
candidate_count = len(final_candidate_ids)
users_per_batch = 8

for start in range(0, len(final_eval_users), users_per_batch):
    batch_users = final_eval_users[start:start + users_per_batch]

    pair_users = np.repeat(batch_users, candidate_count)
    pair_items = np.tile(final_candidate_ids, len(batch_users))

    X_batch = make_pair_features(
        pair_users,
        pair_items,
        final_user_features,
        final_item_features,
        final_profiles,
    )

    scores = final_gbm_model.predict_proba(X_batch)[:, 1].reshape(
        len(batch_users),
        candidate_count,
    )

    for row_idx, user_id in enumerate(batch_users):
        seen = final_seen_by_user.get(user_id, set())

        seen_indices = [
            idx
            for idx, item_id in enumerate(final_candidate_ids)
            if item_id in seen
        ]
        scores[row_idx, seen_indices] = -np.inf

        top_n = min(K, candidate_count)
        top_indices = np.argpartition(
            -scores[row_idx],
            top_n - 1,
        )[:top_n]
        top_indices = top_indices[
            np.argsort(-scores[row_idx][top_indices], kind="stable")
        ]

        final_gbm_recommendations[user_id] = [
            final_candidate_ids[idx] for idx in top_indices
        ]

# Популярность на той же финальной истории и той же выборке пользователей.
final_popular_items = (
    final_train["item_id"].value_counts().index.to_numpy()
)

final_popularity_recommendations = {}

for user_id in final_eval_users:
    seen = final_seen_by_user.get(user_id, set())
    final_popularity_recommendations[user_id] = [
        item_id
        for item_id in final_popular_items
        if item_id not in seen
    ][:K]


def final_metrics(recommendations, users, relevant_by_user, k):
    recalls = []
    ndcgs = []

    for user_id in users:
        relevant = relevant_by_user.get(user_id, set())
        if not relevant:
            continue

        ranked = recommendations.get(user_id, [])[:k]
        hits = sum(item_id in relevant for item_id in ranked)
        recalls.append(hits / len(relevant))

        dcg = sum(
            1 / np.log2(rank + 2)
            for rank, item_id in enumerate(ranked)
            if item_id in relevant
        )
        ideal_hits = min(len(relevant), k)
        idcg = sum(
            1 / np.log2(rank + 2)
            for rank in range(ideal_hits)
        )
        ndcgs.append(dcg / idcg if idcg else 0.0)

    return np.mean(recalls), np.mean(ndcgs)


print(">>> вошли в блок финального теста", flush=True)
print("\n--- Финальный тест: август ---")
print("Пользователей в оценке:", len(final_eval_users))
print(
    "Доля тестовых событий с объектами из каталога:",
    round(
        final_test["item_id"].isin(final_candidate_ids).mean(),
        4,
    ),
)

for name, recs in [
    ("Популярность", final_popularity_recommendations),
    ("Gradient Boosting", final_gbm_recommendations),
]:
    recall, ndcg = final_metrics(
        recs,
        final_eval_users,
        final_relevant_by_user,
        K,
    )
    print(f"{name:20s} Recall@{K}: {recall:.4f}, NDCG@{K}: {ndcg:.4f}")

