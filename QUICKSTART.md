# Быстрый старт

## Установка и запуск

### 1. Установите Rust (если еще не установлен)
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 2. Соберите проект
```bash
cd PIDSettings
cargo build --release
```

### 3. Запустите с примером данных
```bash
cargo run --release -- analyze -i examples/sample_flight.csv
```

## Быстрые команды

### Анализ вашего лога
```bash
# Замените YOUR_LOG.log на путь к вашему файлу
cargo run --release -- analyze -i YOUR_LOG.log -o pid_results.json
```

### Анализ конкретной оси
```bash
# Только Roll
cargo run --release -- analyze -i YOUR_LOG.log -a roll

# Только Pitch
cargo run --release -- analyze -i YOUR_LOG.log -a pitch

# Только Yaw
cargo run --release -- analyze -i YOUR_LOG.log -a yaw

# Только Altitude
cargo run --release -- analyze -i YOUR_LOG.log -a alt
```

### Разные методы настройки
```bash
# Ziegler-Nichols (по умолчанию, хорош для начинающих)
cargo run --release -- analyze -i YOUR_LOG.log -m ziegler-nichols

# Relay Method (более точный)
cargo run --release -- analyze -i YOUR_LOG.log -m relay

# System Identification (адаптивный)
cargo run --release -- analyze -i YOUR_LOG.log -m manual
```

### Экспорт данных в CSV для анализа
```bash
cargo run --release -- extract -i YOUR_LOG.log -o flight_data.csv
```

## Формат лога

Программа принимает логи в двух форматах:

### 1. CSV (самый простой)
Экспортируйте лог из Mission Planner:
1. Откройте Mission Planner
2. Ctrl+F → "Log Browse"
3. Откройте ваш .bin лог
4. Кнопка "Export to CSV"
5. Используйте полученный CSV файл

### 2. .log файл (текстовый формат Ardupilot)
Можете использовать напрямую .log файлы с SD карты.

## Применение результатов

После анализа вы получите что-то вроде:

```
📈 Calculated PID Coefficients:
Method: ziegler-nichols
─────────────────────────────────
Roll:    P: 0.0450, I: 0.0900, D: 0.0056
Pitch:   P: 0.0450, I: 0.0900, D: 0.0056
Yaw:     P: 0.1800, I: 0.0180, D: 0.0000
```

### Как применить:

1. **Умножьте на 0.7 для безопасности:**
   - Roll P: 0.0450 × 0.7 = 0.0315
   - Roll I: 0.0900 × 0.7 = 0.0630
   - Roll D: 0.0056 × 0.7 = 0.0039

2. **Откройте Mission Planner → Config → Full Parameter List**

3. **Найдите и измените параметры:**
   - `ATC_RAT_RLL_P` = 0.0315
   - `ATC_RAT_RLL_I` = 0.0630
   - `ATC_RAT_RLL_D` = 0.0039

4. **Нажмите "Write Params"**

5. **Проведите тестовый полет**

6. **Повторите процесс при необходимости**

## Поиск проблем

### Ошибка: "Insufficient data"
- Лог слишком короткий
- Нужно минимум 10 секунд полета с маневрами

### Ошибка: "Failed to parse log"
- Проверьте формат файла
- Попробуйте экспортировать в CSV из Mission Planner

### Результаты выглядят странно
- Убедитесь, что в логе есть активные маневры
- Полет должен включать движения по всем осям
- Избегайте логов с крашами или аварийными ситуациями

## Советы

1. **Лучшие логи для анализа:**
   - Полет в режиме Stabilize
   - 30-60 секунд активных маневров
   - Плавные движения стиков
   - Без резких ударов и падений

2. **Последовательность настройки:**
   1. Начните с Roll и Pitch (они обычно похожи)
   2. Затем настройте Yaw
   3. В последнюю очередь - Altitude

3. **Признаки хорошей настройки:**
   - Дрон быстро реагирует на команды
   - Нет колебаний при зависании
   - Плавное движение без рывков
   - Стабильное удержание высоты

4. **Признаки плохой настройки:**
   - Дрон "качается" на месте (уменьшите P)
   - Медленная реакция (увеличьте P)
   - Дрон уходит в сторону со временем (увеличьте I)
   - Высокочастотные вибрации (уменьшите D)

## Дополнительно

Полная документация в README.md

GitHub: https://github.com/yourusername/PIDSettings
