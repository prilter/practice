import pandas as pd
import joblib
from train import create_features # Переиспользуем функцию для идентичного преобразования
import os

from consts import *

def main():
    print("Загрузка исходных данных и предобученной модели...")
    original_df = pd.read_csv(DATASETDIR)
    model = joblib.load(MODELDIR)
    
    # Генерация требуемого периода прогнозирования (90 шагов)
    # pd.date_range автоматически учитывает количество дней в месяцах
    forecast_dates = pd.date_range(start='2016-10-03', end='2016-12-31')
    forecast_df = pd.DataFrame({'Date': forecast_dates})
    
    print("Подготовка признаков для горизонта прогноза...")
    X_forecast = create_features(forecast_df)[['Trend', 'DayOfYear', 'DayOfWeek', 'Month']]
    
    print("Выполнение прогнозирования...")
    predictions = model.predict(X_forecast)
    
    # Формируем датафрейм с прогнозом, используя оригинальные названия столбцов из исходника
    columns = original_df.columns.drop('Date')
    predictions_df = pd.DataFrame(predictions, columns=columns)
    
    # Приводим формат дат к строковому, чтобы он совпадал с исходным файлом
    predictions_df.insert(0, 'Date', forecast_dates.strftime('%Y-%m-%d'))
    
    print("Объединение массивов...")
    # Спрогнозированные значения идут строго после исходных, без сброса структуры
    final_df = pd.concat([original_df, predictions_df], ignore_index=True)
    
    # Сохранение итогового результата
    output = AIOUTPUTDIR
    os.makedirs(RESULTDIR, exist_ok=True)
    final_df.to_csv(output, index=False)
    print(f"Готово! Прогноз добавлен. Данные сохранены: {output}")

if __name__ == "__main__":
    main()
