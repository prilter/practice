"""
Практическая работа №2. Классификация данных.
Классификация астрономических объектов (STAR / GALAXY / QSO)
на основе данных Sloan Digital Sky Survey (SDSS).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
# 1. ЗАГРУЗКА И ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ
# ──────────────────────────────────────────────

DATASET_PATH = "public_data.csv"      # исходный датасет
INPUT_PATH = "input.csv"              # файл без целевой переменной (для финальной классификации)
OUTPUT_PATH = "output.csv"            # результат классификации
MODEL_PATH = "model.joblib"           # сохранённая модель
SCALER_PATH = "scaler.joblib"         # сохранённый скейлер
ENCODER_PATH = "label_encoder.joblib" # сохранённый LabelEncoder
TARGET_COLUMN = "class"               # имя целевой переменной

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

# Признаки, которые НЕ несут полезной информации для классификации
# (идентификаторы, служебные поля)
DROP_COLUMNS = [
    "Row_id",
    "obj_ID",
    "run_ID",
    "rerun_ID",
    "cam_col",
    "field_ID",
    "spec_obj_ID",
    "plate",
    "MJD",
    "fiber_ID",
]


def preprocess(df: pd.DataFrame, is_train: bool = True,
               scaler: StandardScaler = None,
               label_encoder: LabelEncoder = None):
    """
    Предобработка данных:
      - удаление неинформативных столбцов;
      - разделение на X и y (если обучающий режим);
      - масштабирование признаков (StandardScaler);
      - кодирование целевой переменной (LabelEncoder).

    Возвращает:
      X_scaled, y_encoded (или None), scaler, label_encoder
    """
    df = df.copy()

    # Удаляем столбцы-идентификаторы (только те, что реально есть в DataFrame)
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # Отделяем целевую переменную (если есть)
    y = None
    if TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN].copy()
        df.drop(columns=[TARGET_COLUMN], inplace=True)

    # Заполняем пропуски медианой (на случай, если они есть)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # ─── Масштабирование ───
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
    else:
        X_scaled = scaler.transform(df)

    # ─── Кодирование меток ───
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
    """
    Обучение нескольких классификаторов, вывод метрик,
    выбор лучшей модели по accuracy.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n────── {name} ──────")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

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
        print(
            classification_report(
                y_test, y_pred, target_names=label_encoder.classes_
            )
        )

    # Определяем лучшую модель
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    print(f"\n★ Лучшая модель: {best_name}  (accuracy = {results[best_name]['accuracy']:.4f})")

    return best_model, best_name, results


def plot_confusion(model, X_test, y_test, label_encoder, model_name: str):
    """Построение и сохранение матрицы ошибок."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Матрица ошибок сохранена в confusion_matrix.png")


def plot_feature_importance(model, feature_names: list, model_name: str):
    """Визуализация важности признаков (для моделей на основе деревьев)."""
    if not hasattr(model, "feature_importances_"):
        print("Модель не поддерживает feature_importances_, пропуск визуализации.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], orient="h")
    plt.title(f"Feature Importance — {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()
    print("График важности признаков сохранён в feature_importance.png")


def plot_results_comparison(results: dict):
    """Сравнительная столбчатая диаграмма метрик по всем моделям."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    model_names = list(results.keys())

    data = {m: [results[n][m] for n in model_names] for m in metrics}
    df_res = pd.DataFrame(data, index=model_names)

    df_res.plot(kind="bar", figsize=(12, 6), rot=15)
    plt.title("Сравнение моделей классификации")
    plt.ylabel("Значение метрики")
    plt.ylim(0.0, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("models_comparison.png", dpi=150)
    plt.show()
    print("Сравнительная диаграмма сохранена в models_comparison.png")


# ──────────────────────────────────────────────
# 4. СОХРАНЕНИЕ / ЗАГРУЗКА МОДЕЛИ
# ──────────────────────────────────────────────

def save_artifacts(model, scaler, label_encoder):
    """Сохранение обученной модели, скейлера и кодировщика на диск."""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    print(f"Артефакты сохранены: {MODEL_PATH}, {SCALER_PATH}, {ENCODER_PATH}")


def load_artifacts():
    """Загрузка модели, скейлера и кодировщика с диска."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("Артефакты загружены.")
    return model, scaler, label_encoder


# ──────────────────────────────────────────────
# 5. КЛАССИФИКАЦИЯ НОВОГО ФАЙЛА (input.csv → output.csv)
# ──────────────────────────────────────────────

def classify_input(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH):
    """
    Считывает input.csv (без целевого столбца),
    применяет обученную модель, записывает output.csv
    (все исходные столбцы + столбец class).
    """
    model, scaler, label_encoder = load_artifacts()

    df_input = pd.read_csv(input_path)
    print(f"Загружен {input_path}: {df_input.shape[0]} строк")

    X, _, _, _, _ = preprocess(df_input, is_train=False, scaler=scaler, label_encoder=label_encoder)

    y_pred_encoded = model.predict(X)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

    df_output = df_input.copy()
    df_output[TARGET_COLUMN] = y_pred_labels

    df_output.to_csv(output_path, index=False)
    print(f"Результат записан в {output_path}")


# ──────────────────────────────────────────────
# 6. ОСНОВНОЙ СЦЕНАРИЙ
# ──────────────────────────────────────────────

def main():
    # ── Шаг 1. Загрузка и анализ ──
    df = load_data(DATASET_PATH)
    explore_data(df)

    # ── Шаг 2. Предобработка ──
    X, y, scaler, label_encoder, feature_names = preprocess(df, is_train=True)

    # ── Шаг 3. Разделение на обучающую и тестовую выборки ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nОбучающая выборка: {X_train.shape[0]} | Тестовая выборка: {X_test.shape[0]}")

    # ── Шаг 4. Обучение и оценка моделей ──
    best_model, best_name, results = train_and_evaluate(
        X_train, y_train, X_test, y_test, label_encoder
    )

    # ── Шаг 5. Визуализации ──
    plot_confusion(best_model, X_test, y_test, label_encoder, best_name)
    plot_feature_importance(best_model, feature_names, best_name)
    plot_results_comparison(results)

    # ── Шаг 6. Кросс-валидация лучшей модели ──
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="accuracy")
    print(f"\nКросс-валидация (5 фолдов): {cv_scores}")
    print(f"Средняя accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Шаг 7. Сохранение артефактов ──
    save_artifacts(best_model, scaler, label_encoder)

    # ── Шаг 8. Классификация input.csv (если файл существует) ──
    if os.path.exists(INPUT_PATH):
        classify_input()
    else:
        print(f"\nФайл {INPUT_PATH} не найден. Пропускаем этап классификации нового файла.")


main()
