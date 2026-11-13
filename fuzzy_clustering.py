import random
import numpy as np
from fuzzy_core import FuzzyNumber
from logger import log

import random
import numpy as np
from fuzzy_core import FuzzyNumber
from logger import log

def generate_fuzzy_dataset(count=100, k_min=2, k_max=150, threshold=0.7):
    """
    [ГСНВ] Генерация N нечетких величин с k от 2 до 150.
    """
    log(f"ГСНВ: Генерация {count} величин (k от {k_min} до {k_max}, порог среза={threshold})...")
    
    # Если запрашиваем до 140 величин — гарантируем уникальные k без повторов!
    pool_k = list(range(k_min, k_max + 1))
    if count <= len(pool_k):
        chosen_k = random.sample(pool_k, count)
        chosen_k.sort()
    else:
        chosen_k = [random.randint(k_min, k_max) for _ in range(count)]

    dataset = []
    for k in chosen_k:
        fn = FuzzyNumber(k=k, threshold_ratio=threshold)
        dataset.append(fn)
    
    log(f"ГСНВ: Успешно создано {len(dataset)} величин.")
    return dataset


def filter_subset_by_relative_width(dataset, alpha_min=0.0, alpha_max=0.5):
    """
    Селекция подмножества по ОТНОСИТЕЛЬНОЙ ширине отрезка (B - A) на шкале от 0.0 до 1.0.
    alpha_min, alpha_max задают долю от минимальной до максимальной ширины в популяции!
    """
    if not dataset:
        return []

    # Находим минимальную и максимальную ширину во всей популяции
    all_widths = [fn.width for fn in dataset]
    w_min = min(all_widths)
    w_max = max(all_widths)
    w_range = w_max - w_min if w_max > w_min else 1.0

    filtered = []
    for fn in dataset:
        # Относительная ширина на шкале [0, 1]
        norm_w = (fn.width - w_min) / w_range
        if alpha_min <= norm_w <= alpha_max:
            filtered.append(fn)

    log(f"Относительный фильтр по ширине [{alpha_min:.2f}, {alpha_max:.2f}]: отобрано {len(filtered)} из {len(dataset)}.")
    return filtered


def cluster_fuzzy_set_by_distance(dataset, n_clusters=3, max_iter=20):
    if len(dataset) == 0:
        return {}, []

    k_clusters = min(n_clusters, len(dataset))
    log(f"КА МСНВ: Кластеризация {len(dataset)} величин на {k_clusters} кластеров (метрика Хаусдорфа)...")

    centroids = random.sample(dataset, k_clusters)
    labels = np.zeros(len(dataset), dtype=int)

    for iteration in range(max_iter):
        new_labels = np.zeros(len(dataset), dtype=int)
        for i, fn in enumerate(dataset):
            distances = [fn.distance(c) for c in centroids]
            new_labels[i] = int(np.argmin(distances))

        if np.array_equal(labels, new_labels) and iteration > 0:
            log(f"КА МСНВ: Сошлось за {iteration} итераций.")
            break
        labels = new_labels

        for c_id in range(k_clusters):
            members = [dataset[i] for i in range(len(dataset)) if labels[i] == c_id]
            if len(members) > 0:
                best_member = members[0]
                min_total_dist = float('inf')
                for cand in members:
                    tot_d = sum(cand.distance(other) for other in members)
                    if tot_d < min_total_dist:
                        min_total_dist = tot_d
                        best_member = cand
                centroids[c_id] = best_member

    clusters = {c_id: [] for c_id in range(k_clusters)}
    for i, c_id in enumerate(labels):
        clusters[c_id].append(dataset[i])

    log(f"КА МСНВ успешно завершен. Распределение: " + 
        ", ".join([f"Кластер {c + 1}: {len(elems)} шт." for c, elems in clusters.items()]))

    return clusters, centroids
