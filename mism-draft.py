import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('centered/full_analysis_results.csv', sep=',')

# df = df[(df['doc_id'] >= 181) & (df['doc_id'] <= 240)]
# ref_df = pd.read_csv('data-full.csv', sep=',')


# Сгруппируем по doc_id и посчитаем количество уникальных diagnosis_type
unique_diagnoses_per_doc = df.groupby('doc_id')['diagnosis_type'].nunique()

# Группируем по количеству уникальных диагнозов
grouped_by_count = {}
for doc_id, count in unique_diagnoses_per_doc.items():
    if count not in grouped_by_count:
        grouped_by_count[count] = []
    grouped_by_count[count].append(doc_id)

# Сортируем по убыванию количества уникальных диагнозов
sorted_counts = sorted(grouped_by_count.keys(), reverse=True)

# Выводим группы
for i, count in enumerate(sorted_counts[:5], 1):  # первые 5 групп
    if i == 1:
        position = "МАКСИМАЛЬНОЕ"
    elif i == 2:
        position = "предмаксимальное"
    elif i == 3:
        position = "пред-предмаксимальное"
    else:
        position = f"{i}-е место"
    
    print(f"{position} количество уникальных диагнозов: {count}")
    print(f"Количество документов: {len(grouped_by_count[count])}")
    print(f"doc_id: {grouped_by_count[count]}")
    
    # Показать диагнозы для первого документа в группе (для примера)
    if grouped_by_count[count]:
        first_doc = grouped_by_count[count][0]
        diagnoses = df[df['doc_id'] == first_doc]['diagnosis_type'].unique()
        print(f"Пример (doc_id {first_doc}): {sorted(diagnoses)}")
    
    print("=" * 50)


# Создаем гистограмму распределения
plt.figure(figsize=(12, 6))
plt.hist(unique_diagnoses_per_doc, bins=range(1, 9), edgecolor='black', alpha=0.7)
plt.xlabel('Количество уникальных диагнозов')
plt.ylabel('Количество документов')
plt.title('Распределение документов по количеству уникальных диагнозов')
plt.xticks(range(1, 9))
plt.grid(axis='y', alpha=0.3)

# Добавляем аннотации для топ-3 значений
sorted_counts = unique_diagnoses_per_doc.sort_values(ascending=False)
for i, (doc_id, count) in enumerate(sorted_counts.head(3).items(), 1):
    position_text = {1: "1-е (макс)", 2: "2-е", 3: "3-е"}[i]
    plt.annotate(f"{position_text}: doc_id {doc_id}", 
                 xy=(count, 0), xytext=(count, 5 + i*3),
                 arrowprops=dict(arrowstyle="->", color='red'),
                 ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Вывод топ-5 документов
print("Топ-5 документов по количеству уникальных диагнозов:")
print("Ранг | doc_id | Кол-во уникальных диагнозов | Диагнозы")
print("-" * 70)

sorted_docs = unique_diagnoses_per_doc.sort_values(ascending=False)
for i, (doc_id, count) in enumerate(sorted_docs.head(5).items(), 1):
    diagnoses = sorted(df[df['doc_id'] == doc_id]['diagnosis_type'].unique())
    print(f"{i:4} | {doc_id:6} | {count:25} | {diagnoses}")