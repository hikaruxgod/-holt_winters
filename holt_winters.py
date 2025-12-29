import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class HoltWintersModel:
    """
    Реализация мультипликативной модели Хольта-Уинтерса для временных рядов с трендом и сезонностью.
    """
    
    def __init__(self, alpha: float = 0.2, beta: float = 0.1, gamma: float = 0.1, 
                 season_length: int = 4):
        """
        Инициализация модели Хольта-Уинтерса.
        
        Parameters:
        -----------
        alpha : float
            Параметр сглаживания уровня (0 < alpha < 1)
        beta : float
            Параметр сглаживания тренда (0 < beta < 1)
        gamma : float
            Параметр сглаживания сезонности (0 < gamma < 1)
        season_length : int
            Длина сезона (например, 4 для квартальных данных)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.season_length = season_length
        
        # Параметры модели (будут вычислены при обучении)
        self.level = None  # Уровень ряда (a)
        self.trend = None  # Тренд (b)
        self.seasonal = None  # Сезонные коэффициенты (F)
        
        # История значений
        self.fitted_values = None
        self.data = None
        
    def _initialize_components(self, data: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Инициализация начальных значений компонент модели.
        
        Parameters:
        -----------
        data : np.ndarray
            Временной ряд
            
        Returns:
        --------
        Tuple[float, float, np.ndarray]
            Начальный уровень, тренд и сезонные коэффициенты
        """
        n = len(data)
        L = self.season_length
        
        # Инициализация уровня (среднее первого сезона)
        initial_level = np.mean(data[:L])
        
        # Инициализация тренда (разница средних между первым и вторым сезоном)
        if n >= 2 * L:
            initial_trend = (np.mean(data[L:2*L]) - np.mean(data[:L])) / L
        else:
            initial_trend = 0
        
        # Инициализация сезонных коэффициентов
        seasonal = np.zeros(L)
        
        # Вычисляем сезонные коэффициенты как отношение значений к среднему уровню
        for i in range(L):
            season_values = []
            for j in range(i, n, L):
                if j < n:
                    season_values.append(data[j])
            if season_values:
                seasonal[i] = np.mean(season_values) / initial_level if initial_level != 0 else 1
        
        # Нормализация сезонных коэффициентов
        seasonal = seasonal / np.mean(seasonal)
        
        return initial_level, initial_trend, seasonal
    
    def fit(self, data: pd.Series) -> 'HoltWintersModel':
        """
        Обучение модели на временном ряде.
        
        Parameters:
        -----------
        data : pd.Series
            Временной ряд для обучения
            
        Returns:
        --------
        self : HoltWintersModel
        """
        self.data = np.array(data)
        n = len(self.data)
        L = self.season_length
        
        # Инициализация компонент
        level, trend, seasonal = self._initialize_components(self.data)
        
        # Массивы для хранения истории компонент
        levels = [level]
        trends = [trend]
        seasonals = list(seasonal)
        fitted = []
        
        # Итеративное обновление компонент по формулам Хольта-Уинтерса
        for t in range(n):
            # Индекс сезонного коэффициента
            season_idx = t % L
            
            # Прогноз на текущий момент
            if t == 0:
                fitted_value = level
            else:
                fitted_value = (levels[-1] + trends[-1]) * seasonals[t]
            fitted.append(fitted_value)
            
            # Обновление уровня
            new_level = self.alpha * (self.data[t] / seasonals[t]) + \
                       (1 - self.alpha) * (levels[-1] + trends[-1])
            
            # Обновление тренда
            new_trend = self.beta * (new_level - levels[-1]) + \
                       (1 - self.beta) * trends[-1]
            
            # Обновление сезонного коэффициента
            new_seasonal = self.gamma * (self.data[t] / new_level) + \
                          (1 - self.gamma) * seasonals[t]
            
            # Сохранение новых значений
            levels.append(new_level)
            trends.append(new_trend)
            seasonals.append(new_seasonal)
        
        # Сохранение финальных параметров модели
        self.level = levels[-1]
        self.trend = trends[-1]
        self.seasonal = np.array(seasonals[-L:])
        self.fitted_values = np.array(fitted)
        
        return self
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Прогнозирование на заданное количество шагов вперёд.
        
        Parameters:
        -----------
        steps : int
            Количество периодов для прогноза
            
        Returns:
        --------
        np.ndarray
            Прогнозные значения
        """
        if self.level is None:
            raise ValueError("Модель не обучена. Вызовите метод fit() перед прогнозированием.")
        
        forecast = []
        for h in range(1, steps + 1):
            season_idx = (len(self.data) + h - 1) % self.season_length
            forecast_value = (self.level + h * self.trend) * self.seasonal[season_idx]
            forecast.append(forecast_value)
        
        return np.array(forecast)
    
    def get_parameters(self) -> dict:
        """
        Получение параметров модели.
        
        Returns:
        --------
        dict
            Словарь с параметрами модели
        """
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'season_length': self.season_length,
            'level': self.level,
            'trend': self.trend,
            'seasonal_coefficients': self.seasonal
        }
    
    def calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> dict:
        """
        Вычисление метрик качества прогноза.
        
        Parameters:
        -----------
        actual : np.ndarray
            Фактические значения
        predicted : np.ndarray
            Прогнозные значения
            
        Returns:
        --------
        dict
            Словарь с метриками (MAE, RMSE, MAPE)
        """
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }


def tune_hyperparameters(data: pd.Series, 
                        alpha_range: Tuple[float, float] = (0.01, 0.99),
                        beta_range: Tuple[float, float] = (0.01, 0.99),
                        gamma_range: Tuple[float, float] = (0.01, 0.99),
                        season_lengths: List[int] = [4],
                        n_trials: int = 50) -> Tuple[dict, float]:
    """
    Подбор оптимальных гиперпараметров методом случайного поиска.
    
    Parameters:
    -----------
    data : pd.Series
        Временной ряд для обучения
    alpha_range : Tuple[float, float]
        Диапазон значений для параметра alpha
    beta_range : Tuple[float, float]
        Диапазон значений для параметра beta
    gamma_range : Tuple[float, float]
        Диапазон значений для параметра gamma
    season_lengths : List[int]
        Список возможных длин сезона
    n_trials : int
        Количество итераций случайного поиска
        
    Returns:
    --------
    Tuple[dict, float]
        Лучшие параметры и соответствующее значение RMSE
    """
    print("Начинается подбор гиперпараметров...")
    print(f"Количество итераций: {n_trials}")
    print()
    
    best_rmse = float('inf')
    best_params = None
    
    # Разделение данных на обучающую и валидационную выборки
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    for i in range(n_trials):
        # Случайный выбор параметров
        alpha = np.random.uniform(*alpha_range)
        beta = np.random.uniform(*beta_range)
        gamma = np.random.uniform(*gamma_range)
        season_length = np.random.choice(season_lengths)
        
        try:
            # Обучение модели
            model = HoltWintersModel(alpha, beta, gamma, season_length)
            model.fit(train_data)
            
            # Прогноз на валидационную выборку
            forecast = model.predict(steps=len(val_data))
            
            # Вычисление RMSE
            rmse = np.sqrt(np.mean((val_data.values - forecast) ** 2))
            
            # Обновление лучших параметров
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {
                    'alpha': alpha,
                    'beta': beta,
                    'gamma': gamma,
                    'season_length': season_length
                }
                print(f"Итерация {i+1}: Новый лучший RMSE = {rmse:.4f}")
                print(f"  Параметры: alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}, L={season_length}")
        
        except Exception as e:
            continue
    
    print()
    print("="*60)
    print("Подбор гиперпараметров завершён!")
    print(f"Лучший RMSE: {best_rmse:.4f}")
    print(f"Лучшие параметры:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print("="*60)
    print()
    
    return best_params, best_rmse


def plot_results(data: pd.Series, fitted: np.ndarray, forecast: np.ndarray, 
                title: str = "Модель Хольта-Уинтерса"):
    """
    Визуализация результатов модели.
    
    Parameters:
    -----------
    data : pd.Series
        Исходные данные
    fitted : np.ndarray
        Подогнанные значения
    forecast : np.ndarray
        Прогнозные значения
    title : str
        Заголовок графика
    """
    plt.figure(figsize=(14, 7))
    
    # Исходные данные
    plt.plot(range(len(data)), data.values, 'o-', label='Исходные данные', 
             linewidth=2, markersize=6)
    
    # Подогнанные значения
    plt.plot(range(len(fitted)), fitted, 's--', label='Подогнанные значения', 
             linewidth=2, markersize=4, alpha=0.7)
    
    # Прогноз
    forecast_index = range(len(data), len(data) + len(forecast))
    plt.plot(forecast_index, forecast, '^-', label='Прогноз', 
             linewidth=2, markersize=8, color='red')
    
    plt.xlabel('Период', fontsize=12)
    plt.ylabel('Значение', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt


def main():
    """
    Основная функция программы.
    """
    print("="*60)
    print("МОДЕЛЬ ХОЛЬТА-УИНТЕРСА")
    print("="*60)
    print()
    
    # Пример данных: квартальные продажи (с трендом и сезонностью)
    # Данные взяты как пример временного ряда с явной сезонностью
    data = pd.Series([
        362, 385, 432, 341,  # Год 1
        382, 409, 498, 387,  # Год 2
        473, 513, 582, 474,  # Год 3
        544, 582, 681, 557,  # Год 4
        628, 707, 773, 592,  # Год 5
    ])
    
    print(f"Длина временного ряда: {len(data)} наблюдений")
    print(f"Данные: {data.values}")
    print()
    
    # Определение количества периодов для прогноза
    if len(data) >= 12:
        forecast_periods = 4  # 2 периода (по 4 квартала = 2 года)
    elif len(data) >= 7:
        forecast_periods = 4  # 1 период (4 квартала = 1 год)
    else:
        forecast_periods = 1
    
    print(f"Количество периодов для прогноза: {forecast_periods}")
    print()
    
    # Выбор режима работы
    print("Выберите режим работы:")
    print("1. Использовать заданные параметры")
    print("2. Выполнить тюнинг гиперпараметров")
    
    # Для демонстрации используем режим с тюнингом
    mode = 2
    
    if mode == 2:
        # Тюнинг гиперпараметров
        print()
        print("Выбран режим: Тюнинг гиперпараметров")
        print()
        
        best_params, best_rmse = tune_hyperparameters(
            data,
            alpha_range=(0.01, 0.99),
            beta_range=(0.01, 0.99),
            gamma_range=(0.01, 0.99),
            season_lengths=[4, 12],  # Тестируем квартальную и месячную сезонность
            n_trials=100
        )
        
        # Обучение модели с лучшими параметрами
        model = HoltWintersModel(**best_params)
    else:
        # Использование заданных параметров
        print()
        print("Выбран режим: Заданные параметры")
        print()
        
        alpha = 0.2
        beta = 0.1
        gamma = 0.3
        season_length = 4
        
        print(f"Параметры модели:")
        print(f"  alpha (уровень): {alpha}")
        print(f"  beta (тренд): {beta}")
        print(f"  gamma (сезонность): {gamma}")
        print(f"  Длина сезона: {season_length}")
        print()
        
        model = HoltWintersModel(alpha, beta, gamma, season_length)
    
    # Обучение модели на всех данных
    print("Обучение модели на полном датасете...")
    model.fit(data)
    print("Модель обучена!")
    print()
    
    # Получение параметров модели
    params = model.get_parameters()
    
    print("="*60)
    print("ПАРАМЕТРЫ МОДЕЛИ ХОЛЬТА-УИНТЕРСА")
    print("="*60)
    print()
    print("Параметры сглаживания:")
    print(f"  α (alpha) - уровень:      {params['alpha']:.4f}")
    print(f"  β (beta) - тренд:         {params['beta']:.4f}")
    print(f"  γ (gamma) - сезонность:   {params['gamma']:.4f}")
    print(f"  L - длина сезона:         {params['season_length']}")
    print()
    
    print("Коэффициенты модели:")
    print(f"  a (уровень):              {params['level']:.4f}")
    print(f"  b (тренд):                {params['trend']:.4f}")
    print()
    
    print("Коэффициенты сезонности (F):")
    for i, coef in enumerate(params['seasonal_coefficients'], 1):
        print(f"  F[{i}]:                      {coef:.4f}")
    print()
    
    # Прогнозирование
    forecast = model.predict(steps=forecast_periods)
    
    print("="*60)
    print("ПРОГНОЗ")
    print("="*60)
    print()
    print(f"Прогноз на {forecast_periods} период(ов):")
    for i, value in enumerate(forecast, 1):
        print(f"  Период {len(data) + i}: {value:.2f}")
    print()
    
    # Метрики качества на обучающей выборке
    metrics = model.calculate_metrics(data.values, model.fitted_values)
    
    print("="*60)
    print("МЕТРИКИ КАЧЕСТВА (на обучающей выборке)")
    print("="*60)
    print()
    print(f"MAE (средняя абсолютная ошибка):     {metrics['MAE']:.4f}")
    print(f"RMSE (среднеквадратичная ошибка):    {metrics['RMSE']:.4f}")
    print(f"MAPE (средняя абсолютная процентная ошибка): {metrics['MAPE']:.2f}%")
    print()
    
    # Визуализация
    print("Создание визуализации...")
    plot = plot_results(data, model.fitted_values, forecast, 
                       "Модель Хольта-Уинтерса: Квартальные продажи")
    plot.savefig(r'C:\Users\hikaru\Documents\Лабораторные и др\4 курс\Иммитационное моделирование\6\outputs\holt_winters_plot.png', dpi=150, bbox_inches='tight')
    print("График сохранён: holt_winters_plot.png")
    print()
    
    print("="*60)
    print("ПРОГРАММА ЗАВЕРШЕНА")
    print("="*60)


if __name__ == "__main__":
    main()
