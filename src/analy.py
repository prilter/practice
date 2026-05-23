"""
Практическая работа №2. Классификация данных.
Шаг 2: Классификация новых данных (input.csv → output.csv).
"""

from consts import *

import os
import warnings
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# НАСТРОЙКИ ПУТЕЙ
# ──────────────────────────────────────────────

INPUT_PATH    = f"{INPUT}/input.csv"
OUTPUT_PATH   = f"{OUTPUT_DIR}/output.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 1. ЗАГРУЗКА АРТЕФАКТОВ
# ──────────────────────────────────────────────

def load_artifacts():
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]):
        raise FileNotFoundError(
            f"Артефакты не найдены в папке «{OUTPUT_DIR}». "
            "Сначала запустите train.py для обучения модели."
        )

    model         = joblib.load(MODEL_PATH)
    scaler        = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("Артефакты успешно загружены.")
    return model, scaler, label_encoder


# ──────────────────────────────────────────────
# 2. ПРЕДОБРАБОТКА ВХОДНЫХ ДАННЫХ
# ──────────────────────────────────────────────

def preprocess_input(df: pd.DataFrame, scaler: StandardScaler):
    """
    Умная предобработка: берем у скейлера список колонок, на которых он обучался, 
    и оставляем из input.csv только их и строго в таком же порядке.
    """
    # Получаем список признаков, которые модель запомнила при обучении
    expected_features = list(scaler.feature_names_in_)
    
    # Проверяем, нет ли в input.csv нехватки обязательных колонок
    missing = [col for col in expected_features if col not in df.columns]
    if missing:
        raise ValueError(
            f"\n[ОШИБКА ДАННЫХ] В файле {INPUT_PATH} отсутствуют необходимые столбцы!\n"
            f"Потеряны колонки: {missing}\n"
            f"Проверьте, правильно ли вы сохранили CSV файл (разделитель должен быть запятой)."
        )

    # Оставляем только нужные колонки в правильном порядке 
    # (ненужные ID отсеются тут автоматически)
    df_clean = df[expected_features].copy()

    # Заполняем пропуски
    df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)

    # Масштабируем
    X_scaled = scaler.transform(df_clean)
    return X_scaled


# ──────────────────────────────────────────────
# 3. КЛАССИФИКАЦИЯ И СОХРАНЕНИЕ РЕЗУЛЬТАТА
# ──────────────────────────────────────────────

def classify(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Входной файл «{input_path}» не найден.")

    model, scaler, label_encoder = load_artifacts()

    # Читаем входные данные
    df_input = pd.read_csv(input_path)
    print(f"Загружен «{input_path}»: {df_input.shape[0]} строк, {df_input.shape[1]} столбцов")

    # Предобработка
    X = preprocess_input(df_input, scaler)

    # Предсказание
    y_pred_encoded = model.predict(X)
    y_pred_labels  = label_encoder.inverse_transform(y_pred_encoded)

    # Формируем выходной файл
    df_output = df_input.copy()
    # Если столбец class зачем-то остался во входном файле - обновляем его, если нет - создаем
    df_output[TARGET_COLUMN] = y_pred_labels

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_output.to_csv(output_path, index=False)

    print(f"\nКлассификация завершена! Результат записан в «{output_path}»")
    print("\n═══ Распределение предсказанных классов ═══")
    print(df_output[TARGET_COLUMN].value_counts())


# ──────────────────────────────────────────────
# ТОЧКА ВХОДА
# ──────────────────────────────────────────────

if __name__ == "__main__":
    classify()
