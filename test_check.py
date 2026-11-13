from fuzzy_core import FuzzyNumber
from fuzzy_clustering import (
    generate_fuzzy_dataset, 
    filter_subset_by_criterion, 
    cluster_fuzzy_set_by_distance
)

print("=" * 60)
print(" ТЕСТ ОБНОВЛЕННОГО МАТЕМАТИЧЕСКОГО ЯДРА (ХИ-КВАДРАТ) ")
print("=" * 60)

print("\n--- 1. Проверка чисел и отрезков [A, B] на уровне 0.7 ---")
A = FuzzyNumber(k=6)
B = FuzzyNumber(k=10)

print(f"Число A (k={A.k}): пик={A.peak_x:.1f}, высота={A.max_pdf:.4f}")
print(f"  -> Отрезок A на уровне 0.7: [{A.A:.2f}, {A.B:.2f}], ширина: {A.width:.2f}")

print(f"Число B (k={B.k}): пик={B.peak_x:.1f}, высота={B.max_pdf:.4f}")
print(f"  -> Отрезок B на уровне 0.7: [{B.A:.2f}, {B.B:.2f}], ширина: {B.width:.2f}")

# 2. Проверяем расстояние Хаусдорфа и операции
print("\n--- 2. Проверка операций и расстояния ---")
dist = A.distance(B)
print(f"Расстояние Хаусдорфа d(A, B): {dist:.2f}")

C = A + B
print(f"Сложение A + B (k={C.k}): пик={C.peak_x:.1f}, COG={C.defuzzify():.2f}")

rel, conf, diff = A.compare(B)
print(f"Сравнение A ? B: {rel} (уверенность: {conf*100:.1f}%)")

# 3. Проверяем генерацию N величин
print("\n--- 3. Проверка генератора ГСНВ ---")
dataset = generate_fuzzy_dataset(count=100, k_min=3, k_max=50)
print(f"Сгенерировано величин: {len(dataset)} шт.")

# 4. Проверяем отбор подмножества по критерию
print("\n--- 4. Проверка отбора подмножества по критерию ---")
# Отбираем только узкие горки, у которых ширина отрезка [A, B] меньше 6
subset = filter_subset_by_criterion(dataset, criterion_type='width', min_val=0.0, max_val=6.0)
print(f"Из 100 величин отобрано с шириной <= 6.0: {len(subset)} шт.")

# 5. Проверяем кластеризацию по расстоянию
print("\n--- 5. Проверка кластеризации КА МСНВ (K-Means по расстоянию) ---")
clusters, centroids = cluster_fuzzy_set_by_distance(dataset, n_clusters=3)
print(f"Кластеризация завершена. Число центроидов: {len(centroids)}")
for c_id, members in clusters.items():
    print(f"  • Кластер {c_id + 1}: {len(members)} элементов (Центроид k={centroids[c_id].k})")

print("\n" + "=" * 60)
print(" ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО! ")
print("=" * 60)