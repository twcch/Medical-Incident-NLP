"""步驟 05 的 IDT 基準：TF-IDF（字元 1–3 gram）＋邏輯迴歸，只用 description，類別平衡權重。

用來回答「LoRA 有沒有比詞袋模型多學到東西」：同一份訓練資料、同樣只看描述、同樣在驗證集調門檻。
需要 scikit-learn；只在 05 的 evaluate 用到，所以在函式內才 import。
"""

from utils.preprocessing import LABEL_COLUMN, TEXT_COLUMN, VALID_IDT_LABELS

TFIDF_SETTINGS = {
    "vectorizer": {"analyzer": "char", "ngram_range": [1, 3], "min_df": 2, "sublinear_tf": True},
    "classifier": {"C": 10.0, "class_weight": "balanced", "max_iter": 5000},
}


def tfidf_scores(
    train_records: list[dict], eval_sets: dict[str, list[dict]]
) -> dict[str, list[float]]:
    """以 train_records 訓練，回傳 {集合名稱: 每筆的 P(系統)}；順序與 eval_sets 裡的紀錄相同。"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError(
            "TF-IDF 基準需要 scikit-learn：pip install scikit-learn；或加 --no-baseline 略過。"
        ) from exc
    system = VALID_IDT_LABELS[1]
    targets = [int(record[LABEL_COLUMN] == system) for record in train_records]
    if len(set(targets)) < 2:
        raise ValueError("訓練資料只有單一類別，無法訓練 TF-IDF 基準。")
    options = TFIDF_SETTINGS["vectorizer"]
    vectorizer = TfidfVectorizer(
        analyzer=options["analyzer"],
        ngram_range=tuple(options["ngram_range"]),
        min_df=options["min_df"],
        sublinear_tf=options["sublinear_tf"],
    )
    features = vectorizer.fit_transform([record[TEXT_COLUMN] for record in train_records])
    classifier = LogisticRegression(**TFIDF_SETTINGS["classifier"]).fit(features, targets)
    return {
        name: classifier.predict_proba(
            vectorizer.transform([record[TEXT_COLUMN] for record in records])
        )[:, 1].tolist()
        for name, records in eval_sets.items()
    }
