# SystemMonitorAgent Separated

Исследовательский Python-проект для восстановления и проверки нейросетевого агента мониторинга состояния вычислительного узла в многоагентной надстройке операционной системы.

## 1. Назначение проекта

Проект моделирует работу интеллектуального агента `SystemMonitorAgent`, который анализирует окно системных метрик, прогнозирует ближайшее будущее состояние узла и формирует диагностический JSON-ответ: текущий класс нагрузки, прогнозируемый класс нагрузки, вероятное узкое место и рекомендуемый статус.

## 2. Архитектура SystemMonitorAgent

В проекте используется разделенная архитектура из двух нейросетевых блоков:

1. `MetricsForecaster` получает окно текущих метрик `[window_size, num_features]` и прогнозирует будущие метрики `[forecast_horizon, num_features]`.
2. `LoadClassifier` классифицирует состояние нагрузки по окну метрик.

Классификатор применяется в двух режимах:

- `CurrentLoadClassifier`: классифицирует текущее состояние системы по реальному текущему окну.
- `FutureLoadClassifier`: классифицирует будущее состояние системы по окну будущих метрик, предсказанных forecaster-моделью.

Итоговая runtime-цепочка:

```text
current metrics window
        |
        +--> CurrentLoadClassifier --> current_load_class
        |
        +--> MetricsForecaster --> forecasted future metrics
                                      |
                                      +--> FutureLoadClassifier --> future_load_class
```

## 3. Почему используется separated architecture

Разделение прогноза метрик и классификации нагрузки делает поведение агента более интерпретируемым. Вместо прямого предсказания будущего класса модель сначала строит прогноз системных показателей, а затем отдельный классификатор оценивает риск:

- можно отдельно сравнивать качество прогнозирования метрик;
- можно отдельно сравнивать качество классификации текущей и будущей нагрузки;
- можно показать, какие прогнозируемые метрики привели к статусу `CRITICAL_RISK_PREDICTED` или `PREVENTIVE_ACTION_REQUIRED`;
- можно сравнить end-to-end цепочку с baseline `future=current`.

## 4. MetricsForecaster

Реализованы четыре модели прогнозирования:

- `MLPForecaster`;
- `CNN1DForecaster`;
- `GRUForecaster`;
- `LSTMForecaster`.

Все модели принимают вход `[batch, window_size, num_features]` и возвращают прогноз `[batch, forecast_horizon, num_features]`. Во время обучения используется weighted MSE, где больший вес получают CPU, память, PSI и blocked processes.

## 5. CurrentLoadClassifier

`CurrentLoadClassifier` получает реальное окно последних метрик и определяет текущий класс нагрузки:

- `NORMAL`;
- `MEDIUM`;
- `HIGH`;
- `CRITICAL`.

Для сравнения обучаются `MLPClassifier`, `CNN1DClassifier`, `GRUClassifier` и `LSTMClassifier`.

## 6. FutureLoadClassifier

`FutureLoadClassifier` обучается на реальных будущих окнах synthetic dataset, но в separated pipeline получает не реальные будущие метрики, а прогноз `MetricsForecaster`. Это важно: end-to-end оценка проверяет именно связку моделей, а не классификацию идеального будущего.

## 7. Формат данных

Файл `data/raw/synthetic_metrics.csv` содержит episode-based временные ряды. В каждой строке есть:

- `episode_id`;
- `scenario_type`;
- `timestep`;
- 12 системных метрик;
- `load_score`;
- `load_class`;
- `future_load_score`;
- `future_load_class`.

Список признаков:

- `cpu_percent`;
- `mem_percent`;
- `swap_percent`;
- `io_read_mb`;
- `io_write_mb`;
- `net_in_mb`;
- `net_out_mb`;
- `psi_cpu`;
- `psi_mem`;
- `psi_io`;
- `process_count`;
- `blocked_processes`.

## 8. Сценарии synthetic dataset

Генератор создает восемь типов эпизодов:

- `normal`: низкая или умеренная стабильная нагрузка;
- `cpu_spike`: короткие всплески CPU;
- `memory_leak`: постепенный рост памяти и swap;
- `io_burst`: периодические всплески I/O и `psi_io`;
- `mixed_overload`: совместный рост CPU, памяти, I/O и PSI;
- `recovery`: высокая нагрузка с последующим снижением;
- `pre_critical_growth`: текущее состояние еще не критическое, но горизонт прогноза переходит в `CRITICAL`;
- `critical_overload`: устойчивое критическое состояние.

PSI-метрики генерируются с задержкой относительно CPU, memory и I/O, чтобы данные были ближе к поведению реальных систем.

## 9. Как запустить

```bash
pip install -r requirements.txt
python src/data_generation.py
python src/train_forecasters.py
python src/train_classifiers.py
python src/evaluate_separated_pipeline.py
python src/system_monitor_agent.py
python src/plot_publication_model_comparison.py
```

Скрипты рассчитаны на запуск из корня проекта `system-monitor-agent-separated`. Пути строятся через `pathlib.Path`, поэтому проект должен работать на Windows и Linux.