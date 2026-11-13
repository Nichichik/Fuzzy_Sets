from nicegui import ui
import matplotlib.pyplot as plt
import numpy as np

from fuzzy_core import FuzzyNumber
from fuzzy_clustering import (
    generate_fuzzy_dataset, 
    filter_subset_by_relative_width, 
    cluster_fuzzy_set_by_distance
)
from logger import log

CLUSTER_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


ui.page_title = "Нечеткая логика: Распределение Хи-квадрат"

ui.label('Комплексное практическое занятие: Нечеткие величины (Хи-квадрат, отрезки [A, B])').classes(
    'text-2xl font-bold text-center w-full text-indigo-700 my-4'
)

with ui.tabs().classes('w-full') as tabs:
    tab_math = ui.tab('1. Манипулирование величинами (+, -, *, /, Сравнение, AND, OR, Расстояние)')
    tab_clusters = ui.tab('2. Генератор, Подмножества и Кластеризация')

with ui.tab_panels(tabs, value=tab_math).classes('w-full p-4'):
    with ui.tab_panel(tab_math):
        with ui.row().classes('w-full gap-6 items-start'):
            
            with ui.card().classes('w-96 p-4 shadow-lg'):
                ui.label('Параметры нечетких чисел').classes('text-lg font-bold mb-2')
                
                ui.label('Число A (степени свободы k):').classes('font-semibold text-blue-600')
                k_a = ui.number('Параметр k (для числа A)', value=6, min=2, max=150, step=1).classes('w-full')
                
                ui.separator().classes('my-2')
                
                ui.label('Число B (степени свободы k):').classes('font-semibold text-green-600')
                k_b = ui.number('Параметр k (для числа B)', value=10, min=2, max=150, step=1).classes('w-full')
                
                ui.separator().classes('my-2')
                
                ui.label('Выберите операцию:').classes('font-semibold')
                operation_select = ui.select(
                    options=[
                        '+ (Сложение)', 
                        '- (Вычитание)', 
                        '* (Умножение)', 
                        '/ (Деление)',
                        'Сравнение (A ? B)',
                        'Расстояние Хаусдорфа d(A, B)',
                        'Пересечение (Fuzzy AND)',
                        'Объединение (Fuzzy OR)'
                    ], 
                    value='+ (Сложение)'
                ).classes('w-full mb-4')
                
                calc_btn = ui.button('Вычислить и отобразить', icon='calculate', color='primary').classes('w-full')
                res_label = ui.markdown('**Результат:** ожидание расчета...').classes('mt-4 text-sm')

            with ui.card().classes('flex-grow p-4 shadow-lg'):
                ui.label('График взаимодействия нечетких величин').classes('text-lg font-bold mb-2')
                plot_area = ui.pyplot(figsize=(9, 5))

        def execute_calculation():
            try:
                op_raw = operation_select.value
                op = op_raw.split()[0]
                log(f"GUI: Расчет операции: {op_raw}...")
                
                A = FuzzyNumber(k=int(k_a.value), compute_all=True)
                B = FuzzyNumber(k=int(k_b.value), compute_all=True)
                
                cog_a = A.defuzzify()
                cog_b = B.defuzzify()
                
                max_x = max(A.B, B.B) * 2.5
                xs = np.linspace(0.001, max(30.0, max_x), 600)
                ys_a = [A.pdf(x) for x in xs]
                ys_b = [B.pdf(x) for x in xs]

                if op == 'Расстояние':
                    dist = A.distance(B)
                    res_label.set_content(
                        f"**Метрическое расстояние между величинами:**<br>"
                        f"• Расстояние Хаусдорфа по срезам [A, B]: <span class='text-lg font-bold text-indigo-600'>{dist:.2f}</span><br>"
                        f"• Отрезок A (срез {A.threshold_ratio:.2f} от пика): **[{A.A:.2f}, {A.B:.2f}]** (высота: {A.cutoff_level:.3f})<br>"
                        f"• Отрезок B (срез {B.threshold_ratio:.2f} от пика): **[{B.A:.2f}, {B.B:.2f}]** (высота: {B.cutoff_level:.3f})"
                    )
                    with plot_area:
                        plt.clf()
                        ax = plt.gca()
                        ax.plot(xs, ys_a, color='#1f77b4', linewidth=2, label=f'A (k={A.k}, пик={A.max_pdf:.3f})')
                        ax.fill_between(xs, 0, ys_a, color='#1f77b4', alpha=0.15)
                        ax.plot(xs, ys_b, color='#2ca02c', linewidth=2, label=f'B (k={B.k}, пик={B.max_pdf:.3f})')
                        ax.fill_between(xs, 0, ys_b, color='#2ca02c', alpha=0.15)
                        
                        ax.hlines(A.cutoff_level, A.A, A.B, colors='#1f77b4', linewidth=4, label=f'Срез A ({A.cutoff_level:.3f})')
                        ax.hlines(B.cutoff_level, B.A, B.B, colors='#2ca02c', linewidth=4, label=f'Срез B ({B.cutoff_level:.3f})')
                        ax.scatter([A.A, A.B], [A.cutoff_level, A.cutoff_level], color='#1f77b4', s=30)
                        ax.scatter([B.A, B.B], [B.cutoff_level, B.cutoff_level], color='#2ca02c', s=30)
                        
                        ax.set_title(f'Расстояние Хаусдорфа d(A, B) = {dist:.2f}')
                        ax.set_xlabel('Базовое множество X')
                        ax.set_ylabel('Функция принадлежности μ(x)')
                        ax.set_ylim(-0.02, 1.05)
                        ax.grid(True, linestyle=':', alpha=0.6)
                        ax.legend()
                        plt.tight_layout()
                    plot_area.update()
                    return

                if op == 'Сравнение':
                    relation, confidence, diff = A.compare(B)
                    res_label.set_content(
                        f"**Результат сравнения величин:**<br>"
                        f"• Вердикт: <span class='text-lg font-bold text-indigo-600'>{relation}</span><br>"
                        f"• Разность центров тяжести: **{diff:+.2f}**<br>"
                        f"• Центр тяжести A = {cog_a:.2f}, Центр тяжести B = {cog_b:.2f}"
                    )
                    with plot_area:
                        plt.clf()
                        ax = plt.gca()
                        ax.plot(xs, ys_a, color='#1f77b4', linewidth=2, label=f'A (k={A.k})')
                        ax.fill_between(xs, 0, ys_a, color='#1f77b4', alpha=0.15)
                        ax.plot(xs, ys_b, color='#2ca02c', linewidth=2, label=f'B (k={B.k})')
                        ax.fill_between(xs, 0, ys_b, color='#2ca02c', alpha=0.15)
                        ax.axvline(cog_a, color='#1f77b4', linestyle=':', label=f'Центр A: {cog_a:.2f}')
                        ax.axvline(cog_b, color='#2ca02c', linestyle=':', label=f'Центр B: {cog_b:.2f}')
                        ax.set_title(f'Сравнение величин: {relation}')
                        ax.set_xlabel('Базовое множество X')
                        ax.set_ylabel('Функция принадлежности μ(x)')
                        ax.set_ylim(-0.02, 1.05)
                        ax.grid(True, linestyle=':', alpha=0.6)
                        ax.legend()
                        plt.tight_layout()
                    plot_area.update()
                    return

                if op in ['Пересечение', 'Объединение']:
                    ys_c = np.minimum(ys_a, ys_b) if op == 'Пересечение' else np.maximum(ys_a, ys_b)
                    sum_y = np.sum(ys_c)
                    cog_c = np.sum(xs * ys_c) / sum_y if sum_y > 0 else 0.0
                    
                    res_label.set_content(f"**Центр тяжести результата ({op}):** **{cog_c:.2f}**")
                    with plot_area:
                        plt.clf()
                        ax = plt.gca()
                        ax.plot(xs, ys_a, color='#1f77b4', linewidth=2, label=f'A (k={A.k})')
                        ax.fill_between(xs, 0, ys_a, color='#1f77b4', alpha=0.15)
                        ax.plot(xs, ys_b, color='#2ca02c', linewidth=2, label=f'B (k={B.k})')
                        ax.fill_between(xs, 0, ys_b, color='#2ca02c', alpha=0.15)
                        ax.plot(xs, ys_c, color='#d62728', linestyle='--', linewidth=2.5, label=f'Результат ({op})')
                        ax.fill_between(xs, 0, ys_c, color='#d62728', alpha=0.25)
                        ax.set_title(f'Операция: {op_raw}')
                        ax.set_xlabel('Базовое множество X')
                        ax.set_ylabel('Функция принадлежности μ(x)')
                        ax.set_ylim(-0.02, 1.05)
                        ax.grid(True, linestyle=':', alpha=0.6)
                        ax.legend()
                        plt.tight_layout()
                    plot_area.update()
                    return

                if op == '+': C = A + B
                elif op == '-': C = A - B
                elif op == '*': C = A * B
                elif op == '/': C = A / B
                
                cog_c = C.defuzzify()
                res_label.set_content(
                    f"**Центры тяжести величин:**<br>"
                    f"• Число A: **{cog_a:.2f}** | Число B: **{cog_b:.2f}** | Результат ({op}): **{cog_c:.2f}**"
                )

                left_c = C.intervals[:, 0]
                right_c = C.intervals[:, 1]
                levels_c = C.levels
                
                xs_c = np.concatenate([[left_c[0]], left_c, right_c[::-1], [right_c[0]]])
                ys_c = np.concatenate([[0.0], levels_c, levels_c[::-1], [0.0]])

                with plot_area:
                    plt.clf()
                    ax = plt.gca()
                    ax.plot(xs, ys_a, color='#1f77b4', linewidth=2, label=f'A (k={A.k}, пик={A.max_pdf:.3f})')
                    ax.fill_between(xs, 0, ys_a, color='#1f77b4', alpha=0.15)
                    ax.plot(xs, ys_b, color='#2ca02c', linewidth=2, label=f'B (k={B.k}, пик={B.max_pdf:.3f})')
                    ax.fill_between(xs, 0, ys_b, color='#2ca02c', alpha=0.15)
                    
                    ax.plot(xs_c, ys_c, color='#d62728', linestyle='--', linewidth=2.5, 
                            label=f'Результат {op} (Центр={cog_c:.2f})')
                    ax.fill(xs_c, ys_c, color='#d62728', alpha=0.25)
                    
                    max_x_view = max(np.max(xs_c) + 4.0, np.max(xs) * 0.7)
                    min_x_view = min(0.0, np.min(xs_c) - 1.0)
                    ax.set_xlim(min_x_view, max_x_view)
                    
                    ax.set_title(f'Операция: {op_raw}')
                    ax.set_xlabel('Базовое множество X')
                    ax.set_ylabel('Функция принадлежности μ(x)')
                    ax.set_ylim(-0.02, 1.05)
                    ax.grid(True, linestyle=':', alpha=0.6)
                    ax.legend()
                    plt.tight_layout()
                plot_area.update()
                
            except Exception as e:
                ui.notify(f"Ошибка: {str(e)}", type='negative')
                log(f"ОШИБКА: {str(e)}")

        calc_btn.on_click(execute_calculation)

 
    with ui.tab_panel(tab_clusters):
        state = {'dataset': [], 'filtered': [], 'clusters': {}, 'centroids': []}

        with ui.row().classes('w-full gap-6 items-start no-wrap'):
            with ui.column().classes('w-[380px] min-w-[380px] gap-4'):
                
                with ui.card().classes('w-full p-4 shadow-lg'):
                    ui.label('1. Генерация величин').classes('text-lg font-bold text-blue-700')
                    count_input = ui.number('Количество величин (N)', value=100, min=20, max=300, step=10).classes('w-full')
                    gen_btn = ui.button('Сгенерировать по k', icon='bolt', color='primary').classes('w-full mt-2')

                with ui.card().classes('w-full p-4 shadow-lg'):
                    ui.label('2. Порог отсечения (Уровень α)').classes('text-lg font-bold text-indigo-700')
                    ui.label('Регулируйте уровень среза от вершины:').classes('text-xs text-gray-500 mb-1')
                    
                    threshold_slider = ui.slider(min=0.0, max=0.95, step=0.01, value=0.7).classes('w-full')
                    threshold_display = ui.label('Текущий порог: 0.70 (70% от пика)').classes('font-bold text-indigo-900')

                with ui.card().classes('w-full p-4 shadow-lg'):
                    ui.label('3. Отбор подмножества').classes('text-lg font-bold text-amber-700')
                    
                    crit_select = ui.select(
                        options=['Все (без фильтра)', 'По относительной ширине отрезка (B - A)'],
                        value='Все (без фильтра)', label='Критерий'
                    ).classes('w-full')
                    
                    ui.label('Диапазон относительной ширины [α_min, α_max]:').classes('text-xs text-gray-500 mt-2')
                    with ui.row().classes('w-full items-center no-wrap gap-2'):
                        alpha_min_input = ui.number('Мин (0..1)', value=0.0, min=0.0, max=1.0, step=0.05).classes('w-1/2')
                        alpha_max_input = ui.number('Макс (0..1)', value=0.5, min=0.0, max=1.0, step=0.05).classes('w-1/2')
                    
                    filter_btn = ui.button('Отобрать подмножество', icon='filter_alt', color='warning').classes('w-full mt-3')

                with ui.card().classes('w-full p-4 shadow-lg'):
                    ui.label('4. Кластерный анализ').classes('text-lg font-bold text-emerald-700')
                    k_clusters_input = ui.number('Количество кластеров', value=3, min=2, max=6, step=1).classes('w-full')
                    cluster_btn = ui.button('Запустить кластеризацию', icon='pie_chart', color='positive').classes('w-full mt-2')

            with ui.column().classes('flex-1 w-full gap-4'):
                with ui.card().classes('w-full p-4 shadow-lg'):
                    ui.label('График нечетких величин').classes('text-lg font-bold mb-2')
                    cluster_plot_area = ui.pyplot(figsize=(12, 5.5))

                with ui.card().classes('w-full p-4 shadow-lg'):
                    ui.label('Таблица параметров нечетких величин').classes('text-md font-bold mb-2')
                    data_table = ui.table(
                        columns=[
                            {'name': 'id', 'label': '№', 'field': 'id', 'sortable': True},
                            {'name': 'k', 'label': 'Параметр k', 'field': 'k', 'sortable': True},
                            {'name': 'peak', 'label': 'Вершина (k-2)', 'field': 'peak', 'sortable': True},
                            {'name': 'A', 'label': 'Левая граница (A)', 'field': 'A', 'sortable': True},
                            {'name': 'B', 'label': 'Правая граница (B)', 'field': 'B', 'sortable': True},
                            {'name': 'width', 'label': 'Ширина (B-A)', 'field': 'width', 'sortable': True},
                            {'name': 'cluster', 'label': 'Кластер', 'field': 'cluster', 'sortable': True},
                        ],
                        rows=[], row_key='id', pagination={'rowsPerPage': 5}
                    ).classes('w-full')
                    cluster_plots_container = ui.column().classes('w-full gap-4 mt-4')

        def update_cluster_view():
            target_data = state['filtered'] if state['filtered'] else state['dataset']
            if not target_data: return

            cluster_map = {}
            if state['clusters']:
                for c_id, members in state['clusters'].items():
                    for m in members:
                        cluster_map[id(m)] = f"Кластер {c_id + 1}"

            rows = []
            for idx, fn in enumerate(target_data):
                rows.append({
                    'id': idx + 1,
                    'k': fn.k,
                    'peak': f"{fn.peak_x:.1f}",
                    'A': f"{fn.A:.2f}",
                    'B': f"{fn.B:.2f}",
                    'width': f"{fn.width:.2f}",
                    'cluster': cluster_map.get(id(fn), "—")
                })
            data_table.rows = rows
            data_table.update()

            with cluster_plot_area:
                plt.clf()
                ax = plt.gca()
                
                item_color = {}
                if state['clusters']:
                    for c_id, members in state['clusters'].items():
                        c_col = CLUSTER_COLORS[c_id % len(CLUSTER_COLORS)]
                        for m in members:
                            item_color[id(m)] = c_col

                total_to_show = len(target_data)
                max_x = max(fn.B for fn in target_data) + 5.0

                for i in range(total_to_show):
                    fn = target_data[i]
                    col = item_color.get(id(fn), '#1f77b4')

                    xs = np.linspace(fn.A, fn.B, 80)
                    ys = [fn.pdf(x) for x in xs]
                    
                    ax.plot(xs, ys, color=col, alpha=0.4, linewidth=1.2)
                    ax.fill_between(xs, fn.cutoff_level, ys, color=col, alpha=0.25)
                    ax.plot([fn.A, fn.B], [fn.cutoff_level, fn.cutoff_level], color=col, linewidth=2.5)
                    ax.scatter([fn.A, fn.B], [fn.cutoff_level, fn.cutoff_level], color=col, s=15)

                cur_ratio = threshold_slider.value
                ax.set_title(f'Усеченные нечеткие величины (срез {cur_ratio*100:.0f}% от пика, основание отсечено)')
                ax.set_xlabel('Базовое множество X')
                ax.set_ylabel('Функция принадлежности μ(x)')
                ax.set_xlim(-1.0, max_x)
                ax.set_ylim(-0.02, 1.05)
                ax.grid(True, linestyle=':', alpha=0.5)
                plt.tight_layout()

            cluster_plot_area.update()

        def on_threshold_change():
            val = float(threshold_slider.value)
            threshold_display.set_text(f'Текущий порог: {val:.2f} ({val*100:.0f}% от пика)')
            
            if state['dataset']:
                for fn in state['dataset']:
                    fn.set_threshold(val)
                
                state['clusters'] = {}
                cluster_plots_container.clear()
                update_cluster_view()
                ui.notify(f'Порог установлен на {val:.2f}! Отрезки [A, B] пересчитаны.', type='info')

        threshold_slider.on('change', on_threshold_change)

        def on_gen():
            cnt = int(count_input.value)
            cur_th = float(threshold_slider.value)
            state['dataset'] = generate_fuzzy_dataset(count=cnt, threshold=cur_th)
            state['filtered'] = state['dataset']
            state['clusters'] = {}
            cluster_plots_container.clear()
            update_cluster_view()
            ui.notify(f'Сгенерировано {cnt} величин с порогом {cur_th:.2f}!', type='positive')

        def on_filter():
            if not state['dataset']: 
                ui.notify('Сначала сгенерируйте величины!', type='warning')
                return

            if crit_select.value == 'По относительной ширине отрезка (B - A)':
                a_min = float(alpha_min_input.value)
                a_max = float(alpha_max_input.value)
                state['filtered'] = filter_subset_by_relative_width(
                    state['dataset'], alpha_min=a_min, alpha_max=a_max
                )
            else:
                state['filtered'] = state['dataset']
            
            state['clusters'] = {}
            cluster_plots_container.clear()
            update_cluster_view()
            ui.notify(f'Отобрано подмножество: {len(state["filtered"])} шт.', type='info')

        def on_cluster():
            target_data = state['filtered'] if state['filtered'] else state['dataset']
            if not target_data: return
            
            k_num = int(k_clusters_input.value)
            clusters, centroids = cluster_fuzzy_set_by_distance(target_data, n_clusters=k_num)
            state['clusters'] = clusters
            state['centroids'] = centroids
            update_cluster_view()

            cluster_plots_container.clear()
            with cluster_plots_container:
                ui.label('Детализация и характеристики кластеров:').classes('text-2xl font-bold text-indigo-900 my-4')

                for c_id, members in clusters.items():
                    c_col = CLUSTER_COLORS[c_id % len(CLUSTER_COLORS)]
                    min_k = min(fn.k for fn in members)
                    max_k = max(fn.k for fn in members)
                    centroid_k = centroids[c_id].k
                    avg_width = np.mean([fn.width for fn in members])
                    avg_center = np.mean([fn.center for fn in members])

                    with ui.card().classes('w-full p-5 shadow-lg border-l-8').style(f'border-color: {c_col}'):
                        with ui.row().classes('w-full justify-between items-center mb-2'):
                            ui.label(f'Кластер {c_id + 1} ({len(members)} величин)').classes('text-xl font-bold').style(f'color: {c_col}')
                            ui.badge(f'Эталонный центроид: k = {centroid_k}', color='indigo').classes('text-sm p-2')
                        
                        ui.markdown(
                            f"• **Диапазон параметров:** k ∈ [{min_k}, {max_k}] &nbsp;|&nbsp; "
                            f"• **Средняя ширина отрезка:** ΔX = {avg_width:.2f} &nbsp;|&nbsp; "
                            f"• **Средний центр на оси X:** X_ср = {avg_center:.2f}"
                        ).classes('text-sm text-gray-700 mb-3')

                        with ui.pyplot(figsize=(10, 3.8)) as sub_plot:
                            ax_sub = sub_plot.fig.gca()
                            local_max_y = max(fn.max_pdf for fn in members)
                            max_x_sub = max(fn.B for fn in members) + 5.0
                            
                            for fn in members:
                                xs = np.linspace(fn.A, fn.B, 80)
                                ys = [fn.pdf(x) for x in xs]
                                ax_sub.plot(xs, ys, color=c_col, alpha=0.45, linewidth=1.2)
                                ax_sub.fill_between(xs, fn.cutoff_level, ys, color=c_col, alpha=0.3)
                                ax_sub.plot([fn.A, fn.B], [fn.cutoff_level, fn.cutoff_level], color=c_col, linewidth=2.5)
                                ax_sub.scatter([fn.A, fn.B], [fn.cutoff_level, fn.cutoff_level], color=c_col, s=15)

                            ax_sub.set_title(f'Кластер {c_id + 1}: усеченные ядра величин')
                            ax_sub.set_xlabel('Базовое множество X', fontsize=10, fontweight='bold')
                            ax_sub.set_ylabel('μ(x)', fontsize=10)
                            ax_sub.set_ylim(-0.01, max(0.3, local_max_y * 1.25))
                            ax_sub.set_xlim(-1.0, max_x_sub)
                            ax_sub.grid(True, linestyle=':', alpha=0.6)
                            sub_plot.fig.tight_layout()

            ui.notify(f'Кластеризация выполнена для уровня {threshold_slider.value:.2f}!', type='positive')

        gen_btn.on_click(on_gen)
        filter_btn.on_click(on_filter)
        cluster_btn.on_click(on_cluster)

if __name__ in {"__main__", "__mp_main__"}:
    log("Запуск обновленного приложения...")
    ui.run(title="Хи-квадрат: Отрезки [A, B]", port=8080, reload=False)