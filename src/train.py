"""
Практическая работа №2. Классификация данных.
Шаг 1: Обучение модели на датасете.
"""

from consts import *

import os
import warnings
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# НАСТРОЙКИ ПУТЕЙ
# ──────────────────────────────────────────────

DATASET_PATH  = f"{INPUT}/public_data.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 1. ЗАГРУЗКА И ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ
# ──────────────────────────────────────────────

DROP_COLUMNS = [
    "Row_id", "obj_ID", "run_ID", "rerun_ID",
    "cam_col", "field_ID", "spec_obj_ID",
    "plate", "MJD", "fiber_ID",
]


def load_data(path: str) -> pd.DataFrame:
    """Загрузка CSV-файла в DataFrame."""
    df = pd.read_csv(path)
    print(f"Загружен файл «{path}»: {df.shape[0]} строк, {df.shape[1]} столбцов")
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Первичный разведочный анализ (EDA)."""
    print("\n═══ Первые 5 строк ═══")
    print(df.head())

    print("\n═══ Информация о датасете ═══")
    print(df.info())

    print("\n═══ Описательная статистика ═══")
    print(df.describe())

    print("\n═══ Пропуски по столбцам ═══")
    print(df.isnull().sum())

    if TARGET_COLUMN in df.columns:
        print("\n═══ Распределение классов ═══")
        print(df[TARGET_COLUMN].value_counts())


# ──────────────────────────────────────────────
# 2. ПРЕДОБРАБОТКА ДАННЫХ
# ──────────────────────────────────────────────

def preprocess(df: pd.DataFrame,
               scaler: StandardScaler = None,
               label_encoder: LabelEncoder = None):
    """Предобработка данных: удаление ненужных столбцов,
    масштабирование, кодирование целевой переменной."""
    df = df.copy()

    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    y = None
    if TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN].copy()
        df.drop(columns=[TARGET_COLUMN], inplace=True)

    df.fillna(df.median(numeric_only=True), inplace=True)

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
    else:
        X_scaled = scaler.transform(df)

    y_encoded = None
    if y is not None:
        if label_encoder is None:
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
        else:
            y_encoded = label_encoder.transform(y)

    return X_scaled, y_encoded, scaler, label_encoder, df.columns.tolist()


# ──────────────────────────────────────────────
# 3. ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ
# ──────────────────────────────────────────────

def train_and_evaluate(X_train, y_train, X_test, y_test, label_encoder):
    """Обучение нескольких классификаторов и выбор лучшего."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n────── {name} ──────")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec  = recall_score(y_test, y_pred, average="weighted")
        f1   = f1_score(y_test, y_pred, average="weighted")

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print()
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    best_name  = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    print(f"\n★ Лучшая модель: {best_name}  (accuracy = {results[best_name]['accuracy']:.4f})")

    return best_model, best_name, results


# ──────────────────────────────────────────────
# 4. ВИЗУАЛИЗАЦИИ
# ──────────────────────────────────────────────

def plot_confusion(model, X_test, y_test, label_encoder, model_name: str):
    """Матрица ошибок."""
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    disp   = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png") 
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Матрица ошибок сохранена в {save_path}")


def plot_feature_importance(model, feature_names: list, model_name: str):
    """Важность признаков (только для деревьев)."""
    if not hasattr(model, "feature_importances_"):
        print("Модель не поддерживает feature_importances_, пропуск.")
        return

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], orient="h")
    plt.title(f"Feature Importance — {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"График важности признаков сохранён в {save_path}")


def plot_results_comparison(results: dict):
    """Сравнительная диаграмма метрик всех моделей."""
    metrics     = ["accuracy", "precision", "recall", "f1"]
    model_names = list(results.keys())

    data   = {m: [results[n][m] for n in model_names] for m in metrics}
    df_res = pd.DataFrame(data, index=model_names)

    df_res.plot(kind="bar", figsize=(12, 6), rot=15)
    plt.title("Сравнение моделей классификации")
    plt.ylabel("Значение метрики")
    plt.ylim(0.0, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "models_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Сравнительная диаграмма сохранена в {save_path}")


# ──────────────────────────────────────────────
# 5. СОХРАНЕНИЕ АРТЕФАКТОВ
# ──────────────────────────────────────────────

def save_artifacts(model, scaler, label_encoder):
    """Сохранение модели, скейлера и энкодера на диск."""
    joblib.dump(model,         MODEL_PATH)
    joblib.dump(scaler,        SCALER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    print(f"\nАртефакты сохранены в папку «{OUTPUT_DIR}»:")
    print(f"  • {MODEL_PATH}")
    print(f"  • {SCALER_PATH}")
    print(f"  • {ENCODER_PATH}")


# ──────────────────────────────────────────────
# ТОЧКА ВХОДА
# ──────────────────────────────────────────────

def main():
    # Шаг 1 — Загрузка и анализ
    df = load_data(DATASET_PATH)
    explore_data(df)

    # Шаг 2 — Предобработка
    X, y, scaler, label_encoder, feature_names = preprocess(df)

    # Шаг 3 — Разделение на train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nОбучающая выборка: {X_train.shape[0]} | Тестовая: {X_test.shape[0]}")

    # Шаг 4 — Обучение и выбор лучшей модели
    best_model, best_name, results = train_and_evaluate(
        X_train, y_train, X_test, y_test, label_encoder
    )

    # Шаг 5 — Кросс-валидация
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="accuracy")
    print(f"\nКросс-валидация (5 фолдов): {cv_scores}")
    print(f"Средняя accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Шаг 6 — Визуализации
    plot_confusion(best_model, X_test, y_test, label_encoder, best_name)
    plot_feature_importance(best_model, feature_names, best_name)
    plot_results_comparison(results)

    # Шаг 7 — Сохранение артефактов
    save_artifacts(best_model, scaler, label_encoder)


main()
