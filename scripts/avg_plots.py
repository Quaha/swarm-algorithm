import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def average_convergence(algo_name, func_id, data_dir="..", steps=20000, grid_step=500):
    build_dir = os.path.join(data_dir, "build")
    pattern = os.path.join(build_dir, f"{algo_name}_{func_id}_run*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found for {algo_name} F{func_id} in {pattern}")
        return None

    # Сетка итераций
    grid = np.arange(0, steps+1, grid_step)
    # Матрица: строки – запуски, столбцы – точки сетки
    all_fitness = []

    for file in files:
        df = pd.read_csv(file)
        # Убедимся, что итерации отсортированы
        df = df.sort_values('Iteration')
        iters = df['Iteration'].values
        fits = df['Fitness'].values

        # Добавляем фиктивную начальную точку (0, fitness на 1-м улучшении? Лучше использовать первое улучшение)
        # Но если первое улучшение после 0, то до этого момента fitness неизвестен. 
        # По логике алгоритма, до первого улучшения fitness равен значению начальной точки.
        # Однако начальная точка в CSV не записывается. Поэтому для интерполяции до первой итерации
        # будем использовать fitness первой точки, а для последней итерации – последний fitness.
        # Создадим интерполятор с экстраполяцией "ближайшим значением"
        interp = interp1d(iters, fits, kind='linear', 
                          fill_value=(fits[0], fits[-1]), bounds_error=False)
        fitness_on_grid = interp(grid)
        all_fitness.append(fitness_on_grid)

    all_fitness = np.array(all_fitness)
    mean_fitness = np.mean(all_fitness, axis=0)
    std_fitness = np.std(all_fitness, axis=0)
    # Доверительный интервал 95%
    ci = 1.96 * std_fitness / np.sqrt(len(files))

    # Сохраняем в CSV
    out_df = pd.DataFrame({
        'Iteration': grid,
        'MeanFitness': mean_fitness,
        'Std': std_fitness,
        'CI_lower': mean_fitness - ci,
        'CI_upper': mean_fitness + ci
    })
    out_df.to_csv(f"{algo_name}_{func_id}_average.csv", index=False)

    # График
    plt.figure(figsize=(10,6))
    plt.plot(grid, mean_fitness, label=f"{algo_name} F{func_id} (mean)", linewidth=2)
    plt.fill_between(grid, mean_fitness-ci, mean_fitness+ci, alpha=0.3, label='95% CI')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title(f'Convergence of {algo_name} on F{func_id} (averaged over {len(files)} runs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{algo_name}_{func_id}_average.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    algorithms = ["SoFA", "SoFAM", "DE", "MINGO", "CRS"]
    functions = [5,1,16,19,28,30]
    for algo in algorithms:
        for f in functions:
            average_convergence(algo, f)