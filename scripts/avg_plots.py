import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ========== CONFIGURATION ==========
STEPS = 20000                # максимальное число итераций
GRID_STEP = 100              # шаг равномерной сетки (чем меньше, тем детальнее)
NUM_RUNS = 30                # ожидаемое число запусков (для проверки)
DATA_DIR = ".."              # папка, содержащая build/
BUILD_DIR = os.path.join(DATA_DIR, "build")
OUTPUT_DIR = "out"           # папка для результатов
SHOW_CI = True               # показывать ли доверительный интервал (95%)
# ===================================

# Список алгоритмов (имена должны совпадать с префиксами в CSV-файлах)
ALGORITHMS = ["SoFA", "SoFAM", "DE", "MINGO", "CRS", "SoFAMG"]

# Список тестовых функций (номера задач CEC17)
FUNCTIONS = [5, 1, 16, 19, 28, 30]

# Необязательные названия функций (для заголовков)
FUNC_NAMES = {
    5: "F5: Rastrigin",
    1: "F1: Bent Cigar",
    16: "F16: Hybrid 6",
    19: "F19: Hybrid 9",
    28: "F28: Composition 8",
    30: "F30: Composition 10"
}


def create_output_dir():
    """Создаёт папку для выходных файлов, если её нет."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_file_pattern(algo, func_id):
    """Возвращает путь к файлам для заданного алгоритма и функции."""
    return os.path.join(BUILD_DIR, f"{algo}_{func_id}_run*.csv")


def load_and_interpolate(algo, func_id, grid):
    """
    Загружает все запуски для (algo, func_id), интерполирует на сетку grid.
    Возвращает:
        - массив значений фитнеса размером (num_runs, len(grid))
        - список имён файлов (для отладки)
    Если файлов нет, возвращает (None, []).
    """
    pattern = get_file_pattern(algo, func_id)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  No files for {algo} F{func_id} in {pattern}")
        return None, []

    print(f"  {algo} F{func_id}: found {len(files)} runs (expected {NUM_RUNS})")

    all_fitness = []
    for file in files:
        df = pd.read_csv(file)
        # Сортируем по итерациям (на всякий случай)
        df = df.sort_values("Iteration")
        iters = df["Iteration"].values
        fits = df["Fitness"].values

        # Создаём интерполятор: линейная интерполяция, экстраполяция первым/последним значением
        # Это корректно: до первого улучшения – значение первого улучшения,
        # после последнего улучшения – последнее значение (алгоритм не ухудшается)
        interp = interp1d(iters, fits, kind="linear",
                          fill_value=(fits[0], fits[-1]), bounds_error=False)
        fitness_on_grid = interp(grid)
        all_fitness.append(fitness_on_grid)

    return np.array(all_fitness), files


def compute_statistics(fitness_matrix):
    """
    Принимает матрицу (runs, grid_points).
    Возвращает среднее, стандартное отклонение, нижнюю и верхнюю границы 95% ДИ.
    """
    mean_fit = np.mean(fitness_matrix, axis=0)
    std_fit = np.std(fitness_matrix, axis=0, ddof=1)
    n = fitness_matrix.shape[0]
    ci = 1.96 * std_fit / np.sqrt(n)   # 95% доверительный интервал для нормального распределения
    lower = mean_fit - ci
    upper = mean_fit + ci
    return mean_fit, std_fit, lower, upper


def plot_function(algo_stats, func_id, grid):
    """
    Рисует один график для заданной функции
    + inset zoom для левого верхнего угла.
    """

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        "SoFA": "#FF1F5B",
        "SoFAM": "#2500F7",
        "DE": "#009ADE",
        "MINGO": "#AF58BA",
        "CRS": "#6AD19E",
        "SoFAMG": "#EC8E00",
    }

    # ========= ОСНОВНОЙ ГРАФИК =========
    for algo, (mean, lower, upper) in algo_stats.items():
        color = colors.get(algo, "#333333")

        ax.plot(grid, mean,
                label=algo,
                color=color,
                linewidth=2.5)

        if SHOW_CI:
            ax.fill_between(grid, lower, upper,
                            color=color,
                            alpha=0.2)

    func_title = FUNC_NAMES.get(func_id, f"Function {func_id}")

    ax.set_title(
        f"Convergence on {func_title}",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Fitness", fontsize=12)

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=10)

    # =========================================================
    # ==================== ZOOM INSET =========================
    # =========================================================

    # Берем первые 35% итераций
    zoom_end = int(STEPS * 0.35)

    # Индексы соответствующего диапазона
    mask = grid <= zoom_end

    # Собираем min/max по всем алгоритмам
    y_values = []

    for algo, (mean, lower, upper) in algo_stats.items():
        y_values.extend(lower[mask])
        y_values.extend(upper[mask])

    y_min = np.min(y_values)
    y_max = np.max(y_values)

    y_min = y_min + (y_max - y_min) * 0.85
    
    print(y_min, y_max)

    # Создаем inset справа сверху
    axins = inset_axes(
        ax,
        width="40%",
        height="40%",
        loc="center right"
    )

    # Рисуем те же данные
    for algo, (mean, lower, upper) in algo_stats.items():
        color = colors.get(algo, "#333333")

        axins.plot(grid, mean,
                   color=color,
                   linewidth=2.0)

        if SHOW_CI:
            axins.fill_between(grid, lower, upper,
                               color=color,
                               alpha=0.2)

    # Границы zoom
    axins.set_xlim(0, zoom_end)
    axins.set_ylim(y_min, y_max)

    # Более компактная сетка
    axins.grid(True, alpha=0.2, linestyle="--")

    # Можно скрыть подписи inset
    axins.tick_params(labelsize=8)

    # Показываем область zoom на основном графике
    mark_inset(
        ax,
        axins,
        loc1=2,
        loc2=4,
        fc="none",
        ec="0.5"
    )

    plt.tight_layout()

    out_file = os.path.join(
        OUTPUT_DIR,
        f"function_{func_id}_average.png"
    )

    plt.savefig(
        out_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"  Saved plot: {out_file}")


def save_average_csv(algo, func_id, grid, mean, lower, upper):
    """Сохраняет усреднённые данные для одного алгоритма в CSV."""
    out_file = os.path.join(OUTPUT_DIR, f"{algo}_{func_id}_average.csv")
    df = pd.DataFrame({
        "Iteration": grid,
        "MeanFitness": mean,
        "CI_lower": lower,
        "CI_upper": upper
    })
    df.to_csv(out_file, index=False)
    print(f"  Saved data: {out_file}")


def main():
    create_output_dir()
    
    # Равномерная сетка итераций
    grid = np.arange(0, STEPS + 1, GRID_STEP)
    
    # Для каждой функции собираем статистику по всем алгоритмам
    for func_id in FUNCTIONS:
        print(f"\nProcessing function F{func_id}...")
        algo_stats = {}   # {algo: (mean, lower, upper)}
        
        for algo in ALGORITHMS:
            fitness_matrix, files = load_and_interpolate(algo, func_id, grid)
            if fitness_matrix is None:
                continue
            
            mean_fit, _, lower, upper = compute_statistics(fitness_matrix)
            algo_stats[algo] = (mean_fit, lower, upper)
            
            # Сохраняем усреднённые данные для дальнейшего использования
            save_average_csv(algo, func_id, grid, mean_fit, lower, upper)
        
        if algo_stats:
            plot_function(algo_stats, func_id, grid)
        else:
            print(f"  No data for function F{func_id}, skipping plot.")
    
    print("\nAll done. Results are in the 'out' folder.")


if __name__ == "__main__":
    main()