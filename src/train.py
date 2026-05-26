import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
import os

from consts import *

def create_features(df):
    """
    Генерирует признаки из дат для ML-модели.
    Алгоритмы не понимают даты напрямую,
    поэтому мы раскладываем их на компоненты.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    
    # Извлечение тренда: количество дней, прошедших с первой даты в датасете
    # Это позволяет модели улавливать общее направление временного ряда
    base_date = pd.to_datetime('2010-01-01') 
    df['Trend'] = (df['Date'] - base_date).dt.days
    
    return df

def main():
    print("Загрузка данных...")
    # Чтение датасета, предоставленного по условию
    df = pd.read_csv(DATASETDIR)
    
    print("Предобработка и извлечение признаков...")
    df_features = create_features(df)
    
    # X - наши признаки, y - целевые переменные (21 временной ряд стран)
    X = df_features[['Trend', 'DayOfYear', 'DayOfWeek', 'Month']]
    y = df.drop(columns=['Date']) 
    
    print("Обучение ML-модели...")
    # MultiOutputRegressor оборачивает базовую модель (Random Forest),
    # обучая отдельный регрессор на каждый из 21 столбцов без смешивания логики.
    base_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(base_model)
    model.fit(X, y)
    
    print("Сохранение модели...")
    os.makedirs(RESULTDIR, exist_ok=True)
    joblib.dump(model, MODELDIR)
    print(f"Модель успешно обучена и сохранена в {MODELDIR}")

if __name__ == "__main__":
    main()
