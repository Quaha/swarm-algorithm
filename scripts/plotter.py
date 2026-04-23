import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob

def plot_single_csv(csv_file, output_file=None, title=None):
    """
    Построение графика зависимости фитнеса от итераций из CSV файла.
    """
    # Чтение CSV файла
    df = pd.read_csv(csv_file)
    
    # Проверка наличия нужных колонок
    if 'Iteration' not in df.columns or 'Fitness' not in df.columns:
        raise ValueError("CSV file must contain 'Iteration' and 'Fitness' columns")
    
    # Создание графика
    plt.figure(figsize=(12, 6))
    
    # Находим максимальную итерацию среди всех данных
    max_iteration = df['Iteration'].max()
    
    # Продлеваем график до правого края
    last_value = df['Fitness'].iloc[-1]
    plt.plot(df['Iteration'], df['Fitness'], 'b-', linewidth=2.0, label='Fitness', color='#FF3366')
    
    # Добавление точек наилучших значений
    best_idx = df['Fitness'].idxmax()
    best_iteration = df['Iteration'][best_idx]
    best_value = df['Fitness'][best_idx]
    
    # Продлеваем линию до правого края если нужно
    if best_iteration < max_iteration:
        plt.hlines(y=best_value, xmin=best_iteration, xmax=max_iteration, 
                  colors='#FF3366', linestyles='b-', linewidth=1.5, alpha=0.7)
    
    plt.plot(best_iteration, best_value, 'r*', 
             markersize=15, label=f'Best: {best_value:.6f}', 
             markeredgecolor='black', markeredgewidth=1)
    
    # Устанавливаем пределы оси X от 0 до max_iteration
    plt.xlim(0, max_iteration)
    
    # Настройка графика
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Fitness Value', fontsize=12)
    
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    else:
        plt.title(f'Convergence Plot - {os.path.basename(csv_file)}', fontsize=14, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Сохранение или отображение
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def plot_multiple_csv(csv_files, labels=None, output_file=None, title=None, 
                      use_log_scale=False, mark_best=True, linewidth=2.0):
    """
    Построение нескольких графиков на одном полотне.
    
    Parameters:
    -----------
    csv_files : list
        Список путей к CSV файлам
    labels : list, optional
        Метки для легенды (по умолчанию имена файлов без расширения)
    output_file : str, optional
        Путь для сохранения графика
    title : str, optional
        Заголовок графика
    use_log_scale : bool
        Использовать логарифмическую шкалу по Y
    mark_best : bool
        Отмечать ли лучшие точки на графиках
    linewidth : float
        Толщина линий
    """
    # Настройка стиля
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Создание графика
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Яркая цветовая схема
    bright_colors = [
        '#FF1F5B',  # Ярко-розовый
        '#00CD6C',  # Ярко-зеленый
        '#009ADE',  # Ярко-голубой
        '#AF58BA',  # Ярко-фиолетовый
        '#FFC61E',  # Ярко-желтый
        '#F28522',  # Ярко-оранжевый
        '#FF3366',  # Красный
        '#00BFFF',  # Глубокий голубой
        '#32CD32',  # Лаймовый
        '#FF69B4',  # Горячий розовый
        '#FFD700',  # Золотой
        '#7FFF00',  # Шартрез
        '#FF4500',  # Оранжево-красный
        '#00FF7F',  # Весенний зеленый
        '#FF1493',  # Глубокий розовый
    ]
    
    # Если цветов не хватает, дополняем из других палитр
    if len(csv_files) > len(bright_colors):
        additional_colors = plt.cm.tab10(np.linspace(0, 1, len(csv_files) - len(bright_colors)))
        colors = bright_colors + [f'#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}' 
                                 for r, g, b, _ in additional_colors]
    else:
        colors = bright_colors[:len(csv_files)]
    
    # Если метки не указаны, используем имена файлов
    if labels is None:
        labels = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
    
    # Подгоняем длину labels к длине csv_files
    if len(labels) < len(csv_files):
        labels.extend([f"File {i+1}" for i in range(len(labels), len(csv_files))])
    
    best_values = []
    max_iteration = 0
    
    # Первый проход: находим максимальную итерацию среди всех файлов
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'Iteration' in df.columns:
                max_iteration = max(max_iteration, df['Iteration'].max())
        except Exception:
            continue
    
    # Построение графиков
    for i, (csv_file, label) in enumerate(zip(csv_files, labels)):
        try:
            df = pd.read_csv(csv_file)
            
            # Проверка колонок
            if 'Iteration' not in df.columns or 'Fitness' not in df.columns:
                print(f"Warning: {csv_file} doesn't have required columns. Skipping.")
                continue
            
            # Построение линии
            line = ax.plot(df['Iteration'], df['Fitness'], 
                          color=colors[i], linewidth=linewidth, 
                          label=label, alpha=0.9)
            
            # Продлеваем линию до правого края
            if len(df) > 0:
                last_iteration = df['Iteration'].iloc[-1]
                last_value = df['Fitness'].iloc[-1]
                
                if last_iteration < max_iteration:
                    # Добавляем горизонтальную линию от последней точки до max_iteration
                    ax.hlines(y=last_value, xmin=last_iteration, xmax=max_iteration, 
                            colors=colors[i], linestyles='solid', linewidth=linewidth*0.7, alpha=0.6)
            
            # Отмечаем лучшую точку
            if mark_best and len(df) > 0:
                best_idx = df['Fitness'].idxmax()
                best_value = df['Fitness'][best_idx]
                best_iteration = df['Iteration'][best_idx]
                
                ax.plot(best_iteration, best_value, '*', 
                       color=colors[i], markersize=15, 
                       markeredgecolor='black', markeredgewidth=1.5)
                
                # Добавляем пунктирную линию от лучшей точки до правого края
                if best_iteration < max_iteration:
                    ax.hlines(y=best_value, xmin=best_iteration, xmax=max_iteration, 
                            colors=colors[i], linestyles='--', linewidth=linewidth*0.5, alpha=0.4)
                
                best_values.append((label, best_value, best_iteration, colors[i]))
                
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    # Устанавливаем пределы оси X
    ax.set_xlim(0, max_iteration)
    
    # Настройка осей
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fitness Value', fontsize=12, fontweight='bold')
    
    # Заголовок
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Convergence Comparison', fontsize=14, fontweight='bold')
    
    # Сетка и легенда
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    
    # Логарифмическая шкала
    if use_log_scale:
        ax.set_yscale('log')
    
    # Добавление аннотации с лучшими значениями
    if best_values:
        # Сортируем по значению фитнеса
        best_values_sorted = sorted(best_values, key=lambda x: x[1])
        
        annotation_text = "Best values:\n"
        for label, value, iteration, color in best_values_sorted:
            annotation_text += f"{label}: {value:.6f} (iter {iteration})\n"
        
        # Размещаем текст в правом нижнем углу
        plt.figtext(0.99, 0.01, annotation_text, 
                   ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='gray'),
                   fontsize=9, family='monospace')
    
    plt.tight_layout()
    
    # Сохранение или отображение
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
        
        # Сохраняем также CSV с лучшими значениями
        stats_file = os.path.splitext(output_file)[0] + '_stats.csv'
        if best_values:
            stats_df = pd.DataFrame([(l, v, i) for l, v, i, _ in best_values], 
                                   columns=['Label', 'Best_Fitness', 'Best_Iteration'])
            stats_df.to_csv(stats_file, index=False)
            print(f"Best values saved to {stats_file}")
    else:
        plt.show()
    
    # Вывод статистики в консоль
    if best_values:
        best_values_sorted = sorted(best_values, key=lambda x: x[1])
        
        print("\n" + "="*60)
        print("BEST VALUES SUMMARY")
        print("="*60)
        for label, value, iteration, _ in best_values_sorted:
            print(f"{label:.<40} {value:.10f} (iter: {iteration})")
        print("="*60)


def plot_csv_with_wildcards(pattern, labels=None, output_file=None, title=None, **kwargs):
    """
    Поддержка wildcards для выбора файлов.
    
    Examples:
    ---------
    plot_csv_with_wildcards("results/*.csv")
    plot_csv_with_wildcards("run_*.csv")
    """
    csv_files = sorted(glob.glob(pattern))
    
    if not csv_files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(csv_files)} files:")
    for f in csv_files:
        print(f"  - {f}")
    
    plot_multiple_csv(csv_files, labels, output_file, title, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot fitness convergence from one or multiple CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python plot_convergence.py results.csv
  
  # Multiple files with custom labels
  python plot_convergence.py run1.csv run2.csv run3.csv -l "GA" "PSO" "DE"
  
  # Using wildcards
  python plot_convergence.py results/*.csv -o comparison.png
  
  # Multiple files with log scale
  python plot_convergence.py run1.csv run2.csv --log
  
  # Disable best point marking
  python plot_convergence.py run1.csv run2.csv --no-mark
        """
    )
    
    parser.add_argument('csv_files', nargs='*', 
                       help='CSV files to plot (supports wildcards like results/*.csv)')
    parser.add_argument('-l', '--labels', nargs='+', 
                       help='Labels for legend')
    parser.add_argument('-o', '--output', 
                       help='Output image file path')
    parser.add_argument('-t', '--title', 
                       help='Plot title')
    parser.add_argument('--log', action='store_true',
                       help='Use logarithmic scale for Y axis')
    parser.add_argument('--no-mark', action='store_false', dest='mark_best',
                       help='Do not mark best points')
    parser.add_argument('--linewidth', type=float, default=2.0,
                       help='Line width for plots (default: 2.0)')
    parser.add_argument('-p', '--pattern',
                       help='File pattern with wildcards (e.g., "results/*.csv")')
    
    args = parser.parse_args()
    
    # Собираем все файлы
    all_files = []
    
    # Добавляем файлы из аргументов
    if args.csv_files:
        for f in args.csv_files:
            # Проверяем, не является ли это wildcard паттерном
            if '*' in f or '?' in f:
                all_files.extend(sorted(glob.glob(f)))
            else:
                all_files.append(f)
    
    # Добавляем файлы из --pattern
    if args.pattern:
        all_files.extend(sorted(glob.glob(args.pattern)))
    
    # Удаляем дубликаты
    all_files = list(dict.fromkeys(all_files))
    
    if not all_files:
        parser.print_help()
        print("\nError: No CSV files specified!")
        exit(1)
    
    print(f"Plotting {len(all_files)} file(s)...")
    
    # Выбор режима: один файл или несколько
    if len(all_files) == 1:
        plot_single_csv(all_files[0], args.output, args.title)
    else:
        plot_multiple_csv(
            all_files, 
            labels=args.labels,
            output_file=args.output,
            title=args.title,
            use_log_scale=args.log,
            mark_best=args.mark_best,
            linewidth=args.linewidth
        )