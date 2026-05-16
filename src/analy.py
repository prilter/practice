''' DATASET DESCRIPION '''

import json
import re
from datetime import datetime
from statistics import mean, median

# FUNCTIONS
def count_words_in_text(text) -> int:
    '''Подсчитывает количество слов в тексте'''
    if not isinstance(text, str) or not text:
        return 0
    words = re.findall(r'\b\w+\b', text, re.UNICODE)
    return len(words)


def get_total_records_count(a):
    ''' Пункт 1: Количество записей '''
    return len(a)


def get_unique_words_count(data, field="title"):
    '''
    Пункт 2: Количество уникальных слов в указанном поле (по умолчанию title)
    '''
    all_words = []
    for item in data:
        text = item.get(field, '')
        if text:
            words = re.findall(r'\b\w+\b', text, re.UNICODE)
            all_words.extend([word.lower() for word in words])
    
    return len(set(all_words))


def get_word_count_statistics(data, field="data"):
    '''
    Пункт 3: Минимальное, максимальное, среднее, медианное количество слов в записях
    '''
    word_counts = [count_words_in_text(item.get(field, '')) for item in data]
    
    if not word_counts:
        return { 'min': 0, 'max': 0, 'mean': 0.0, 'median': 0.0 }
    
    return {
        'min': min(word_counts),
        'max': max(word_counts),
        'mean': mean(word_counts),
        'median': median(word_counts)
    }


def get_date_range(data, date_field="date"):
    '''
    Пункт 4: Диапазон дат опубликования записей
    Поддерживает формат: "Fri, 01 May 2026 12:45:34 GMT"
    '''
    dates = []
    
    for item in data:
        date_str = item.get(date_field, '')
        if date_str:
            try:
                # Check different date formats
                for fmt in ["%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%d", "%d.%m.%Y"]:
                    try:
                        date_obj = datetime.strptime(date_str, fmt)
                        dates.append(date_obj)
                        break
                    except ValueError:
                        continue
            except:
                continue
    
    if not dates:
        return {
            'min_date': None,
            'max_date': None,
            'range_days': None
        }
    
    min_date = min(dates)
    max_date = max(dates)
    range_days = (max_date - min_date).days
    
    return {
        'min_date': min_date.strftime('%Y-%m-%d'),
        'max_date': max_date.strftime('%Y-%m-%d'),
        'range_days': range_days
    }


def get_missing_values_ratio(data, attributes = None):
    '''
    Пункт 5: Доля пропусков в записях по каждому из атрибутов
    Возвращает словарь {атрибут: процент_пропусков}
    '''
    if not data: # No data
        return {}

    if attributes is None:
        attributes = list(data[0].keys()) # Set empty attributes
    
    total_records = len(data)
    missing_ratio = {}
    
    for attr in attributes:
        missing_count = 0
        for item in data:
            value = item.get(attr)
            
            # Check is empty
            if value is None or value == '' or value == []:
                missing_count += 1
        
        missing_ratio[attr] = (missing_count / total_records) * 100 if total_records > 0 else 0
    
    return missing_ratio


# MAIN
def main(fn: str):
    print("=" * 60)
    print("Statistic")
    print("=" * 60)

    # 0. Get data
    data = []
    with open(fn, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Records counter 
    total = get_total_records_count(data)
    print(f"\n1. Количество записей: {total}")
    
    # 2. Uniqie words
    unique_words = get_unique_words_count(data)
    print(f"\n2. Количество уникальных слов (в поле 'date'): {unique_words}")
    
    # 3. Words stats
    word_stats = get_word_count_statistics(data)
    print(f"\n3. Статистика количества слов (в поле 'data'):")
    print(f"   - Минимальное: {word_stats['min']}")
    print(f"   - Максимальное: {word_stats['max']}")
    print(f"   - Среднее: {word_stats['mean']:.2f}")
    print(f"   - Медианное: {word_stats['median']:.2f}")
    
    # 4. Dates range
    date_range = get_date_range(data)
    print(f"\n4. Диапазон дат (поле 'date'):")
    if date_range['min_date']:
        print(f"   - С: {date_range['min_date']}")
        print(f"   - По: {date_range['max_date']}")
        print(f"   - Всего дней: {date_range['range_days']}")
    else:
        print("   - Нет корректных дат для анализа")
    
    # 5. Empty fields stats
    missing_ratios = get_missing_values_ratio(data)
    print(f"\n5. Доля пропусков по атрибутам:")
    for attr, ratio in missing_ratios.items():
        print(f"   - {attr:15}: {ratio:5.2f}%")
    
    print("\n" + "=" * 60)

main(input('Enter file dir: '))
