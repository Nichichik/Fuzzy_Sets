import numpy as np
import scipy.special as sp

class FuzzyNumber:
    """
    Класс нечеткого числа на основе распределения Хи-квадрат.
    Глобально нормализован относительно максимума при k=2 (0.5 -> 1.0).
    Диапазон высоты Y строго [0.0, 1.0]
    """
    def __init__(self, k=5, threshold_ratio=0.7, levels=None, intervals=None, compute_all=False):
        self.k = max(2, int(k))
        self.threshold_ratio = threshold_ratio

        if self.k > 2:
            self.peak_x = float(self.k - 2)
        else:
            self.peak_x = 0.0

        self.max_pdf = self.pdf(self.peak_x)
        
        self.cutoff_level = float(self.threshold_ratio * self.max_pdf)

        self.A, self.B = self._find_single_cut_fast(self.cutoff_level)
        self.width = self.B - self.A
        self.center = (self.A + self.B) / 2.0

        if intervals is not None:
            self.intervals = np.array(intervals)
            self.levels = np.array(levels) if levels is not None else np.linspace(0.005, self.max_pdf * 0.98, len(self.intervals))
        elif compute_all:
            self.intervals, self.levels = self._generate_intervals_fast()
        else:
            self.intervals = np.array([[self.A, self.B]])
            self.levels = np.array([self.cutoff_level])

    def pdf(self, x):
        """Плотность вероятности Хи-квадрат (нормализованная к 1.0 при k=2)"""
        if x < 0:
            return 0.0
        if x == 0:
            return 1.0 if self.k == 2 else 0.0

        try:
            log_coef = -(self.k / 2.0) * np.log(2.0) - sp.gammaln(self.k / 2.0)
            log_power = (self.k / 2.0 - 1.0) * np.log(x)
            log_exp = -x / 2.0
            raw_val = np.exp(log_coef + log_power + log_exp)
            
            # Нормализация через 0.5 (максимум при k=2)
            norm_val = raw_val / 0.5
            return float(min(1.0, max(0.0, norm_val)))
        except:
            return 0.0

    def _find_single_cut_fast(self, y_level):
        """Быстрый поиск точек A и B методом бисекции"""
        if y_level >= self.max_pdf:
            return self.peak_x, self.peak_x

        # Левый край A
        if self.k == 2:
            a = 0.0
        else:
            low, high = 0.0001, self.peak_x
            for _ in range(25):
                mid = (low + high) / 2.0
                if self.pdf(mid) < y_level:
                    low = mid
                else:
                    high = mid
            a = (low + high) / 2.0

        # Правый край B
        low = self.peak_x
        step = max(3.0, np.sqrt(2.0 * self.k))
        high = low + step
        while self.pdf(high) > y_level and high < 1000.0:
            low = high
            high += step

        for _ in range(25):
            mid = (low + high) / 2.0
            if self.pdf(mid) > y_level:
                low = mid
            else:
                high = mid
        b = (low + high) / 2.0

        return float(a), float(b)

    def _generate_intervals_fast(self):
        """Генерация 30 уровней срезов от 0 до верхушки собственного пика"""
        my_levels = np.linspace(0.002, self.max_pdf * 0.98, 30)
        res = [self._find_single_cut_fast(y) for y in my_levels]
        return np.array(res), my_levels

    def defuzzify(self):
        """Центр тяжести по срезам"""
        midpoints = (self.intervals[:, 0] + self.intervals[:, 1]) / 2.0
        return float(np.mean(midpoints))

    def distance(self, other):
        """Расстояние Хаусдорфа по отрезкам [A, B]"""
        return float(max(abs(self.A - other.A), abs(self.B - other.B)))


    def _get_common_cuts(self, other):
        """Строит общую согласованную сетку срезов для двух чисел до меньшего пика"""
        common_max_y = min(self.max_pdf, other.max_pdf) * 0.98
        common_levels = np.linspace(0.002, common_max_y, 30)
        inv_a = [self._find_single_cut_fast(y) for y in common_levels]
        inv_b = [other._find_single_cut_fast(y) for y in common_levels]
        return common_levels, inv_a, inv_b

    def __add__(self, other):
        levels, inv_a, inv_b = self._get_common_cuts(other)
        new_inv = [[a_l + b_l, a_r + b_r] for (a_l, a_r), (b_l, b_r) in zip(inv_a, inv_b)]
        return FuzzyNumber(k=self.k + other.k, levels=levels, intervals=new_inv)

    def __sub__(self, other):
        levels, inv_a, inv_b = self._get_common_cuts(other)
        new_inv = [[a_l - b_r, a_r - b_l] for (a_l, a_r), (b_l, b_r) in zip(inv_a, inv_b)]
        return FuzzyNumber(k=max(2, abs(self.k - other.k)), levels=levels, intervals=new_inv)

    def __mul__(self, other):
        levels, inv_a, inv_b = self._get_common_cuts(other)
        new_inv = []
        for (a_l, a_r), (b_l, b_r) in zip(inv_a, inv_b):
            p = [a_l * b_l, a_l * b_r, a_r * b_l, a_r * b_r]
            new_inv.append([min(p), max(p)])
        return FuzzyNumber(k=self.k, levels=levels, intervals=new_inv)

    def __truediv__(self, other):
        levels, inv_a, inv_b = self._get_common_cuts(other)
        new_inv = []
        for (a_l, a_r), (b_l, b_r) in zip(inv_a, inv_b):
            b_l_safe = max(0.001, b_l)
            b_r_safe = max(0.001, b_r)
            divs = [a_l / b_r_safe, a_l / b_l_safe, a_r / b_r_safe, a_r / b_l_safe]
            new_inv.append([min(divs), max(divs)])
        return FuzzyNumber(k=self.k, levels=levels, intervals=new_inv)

    def compare(self, other):
        diff = self.defuzzify() - other.defuzzify()
        if abs(diff) < 0.1: rel = "A ≈ B"
        elif diff > 0: rel = "A > B"
        else: rel = "A < B"
        return rel, 1.0, diff

    def set_threshold(self, new_ratio):
        """Динамически пересчитывает отрезок [A, B] при движении ползунка пользователем"""
        self.threshold_ratio = float(new_ratio)
        self.cutoff_level = float(self.threshold_ratio * self.max_pdf)
        self.A, self.B = self._find_single_cut_fast(self.cutoff_level)
        self.width = self.B - self.A
        self.center = (self.A + self.B) / 2.0