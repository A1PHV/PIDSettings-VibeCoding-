# PID Tuner для Ardupilot

**Автоматический расчет PID коэффициентов из логов полетов Ardupilot**

Программа для анализа логов Ardupilot и расчета оптимальных PID коэффициентов для наилучшего управления БПЛА.

## 📦 Две версии программы

### 🐍 Python версия (РЕКОМЕНДУЕТСЯ)
- ✅ **Поддержка .bin логов** с SD-карты (DataFlash)
- ✅ **Портативный .exe** без зависимостей
- ✅ **Работает БЕЗ интернета** после сборки
- 📂 **Папка:** `python_version/`
- 📖 **Документация:** [python_version/README.md](python_version/README.md)

### 🦀 Rust версия
- ✅ Поддержка .log и .csv логов
- ✅ Высокая производительность
- ❌ Не поддерживает .bin напрямую
- 📂 **Папка:** `src/` (Rust код)

---

> **💡 Для работы с .bin логами используйте Python версию!**

---

## Возможности

- Парсинг логов Ardupilot (форматы .log, .csv)
- Расчет PID коэффициентов для всех осей (Roll, Pitch, Yaw, Altitude)
- Несколько методов настройки:
  - **Ziegler-Nichols** - классический метод на основе анализа колебаний
  - **Relay Method** - метод Åström-Hägglund с релейной обратной связью
  - **Manual** - идентификация системы методом наименьших квадратов
- Экспорт данных в CSV для дополнительного анализа
- Генерация параметров в формате Ardupilot

## Установка

### Требования
- Rust 1.70 или выше
- Cargo (входит в состав Rust)

### Установка Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Сборка проекта
```bash
# Клонируйте репозиторий
git clone <repository-url>
cd PIDSettings

# Соберите проект
cargo build --release

# Исполняемый файл будет в target/release/pid_tuner
```

## Использование

### Анализ логов и расчет PID коэффициентов

```bash
# Анализ всех осей методом Ziegler-Nichols
cargo run --release -- analyze -i flight_log.log

# Анализ конкретной оси
cargo run --release -- analyze -i flight_log.log -a roll

# Использование другого метода
cargo run --release -- analyze -i flight_log.log -m relay

# Сохранение результатов в файл
cargo run --release -- analyze -i flight_log.log -o pid_results.json
```

### Извлечение данных в CSV

```bash
cargo run --release -- extract -i flight_log.log -o flight_data.csv
```

### Параметры

**Команда `analyze`:**
- `-i, --input <FILE>` - Путь к лог-файлу Ardupilot
- `-o, --output <FILE>` - Файл для сохранения результатов (JSON)
- `-a, --axis <AXIS>` - Ось для настройки: roll, pitch, yaw, alt, all (по умолчанию: all)
- `-m, --method <METHOD>` - Метод расчета: ziegler-nichols, relay, manual (по умолчанию: ziegler-nichols)

**Команда `extract`:**
- `-i, --input <FILE>` - Путь к лог-файлу Ardupilot
- `-o, --output <FILE>` - Файл для сохранения данных (CSV)

## Методы настройки PID

### 1. Ziegler-Nichols (рекомендуется для начинающих)
Классический метод, основанный на анализе характеристик колебаний системы:
- **P** = 0.6 × Ku (предельное усиление)
- **I** = 2P / Tu (период колебаний)
- **D** = P × Tu / 8

### 2. Relay Method (для точной настройки)
Метод с релейной обратной связью, позволяет найти точку автоколебаний:
- Более консервативный подход
- Лучше подходит для нестабильных систем

### 3. Manual (системная идентификация)
Использует метод наименьших квадратов для оценки параметров системы:
- Учитывает перерегулирование и время переходного процесса
- Адаптируется к характеру полета

## Формат логов

Программа поддерживает два формата:

### 1. Нативный формат .log Ardupilot
```
ATT, TimeUS, DesRoll, Roll, DesPitch, Pitch, DesYaw, Yaw
RATE, TimeUS, RDes, R, PDes, P, YDes, Y
CTUN, TimeUS, DAlt, Alt, ...
```

### 2. CSV формат (экспорт из Mission Planner/MAVExplorer)
CSV с заголовками, содержащий поля:
- timestamp/time/TimeUS
- Roll/roll, DesRoll/roll_des
- Pitch/pitch, DesPitch/pitch_des
- Yaw/yaw, DesYaw/yaw_des
- Alt/alt, DAlt/alt_des

## Применение результатов

### Просмотр результатов
После анализа программа выведет:
```
📈 Calculated PID Coefficients:
Method: ziegler-nichols
─────────────────────────────────
Roll:    P: 0.0450, I: 0.0900, D: 0.0056
Pitch:   P: 0.0450, I: 0.0900, D: 0.0056
Yaw:     P: 0.1800, I: 0.0180, D: 0.0000
Altitude: P: 1.0000, I: 0.5000, D: 0.0000
```

### Применение в Mission Planner

1. Откройте Mission Planner
2. Подключитесь к контроллеру полета
3. Перейдите в **Config → Full Parameter List**
4. Найдите соответствующие параметры:
   - **Roll**: `ATC_RAT_RLL_P`, `ATC_RAT_RLL_I`, `ATC_RAT_RLL_D`
   - **Pitch**: `ATC_RAT_PIT_P`, `ATC_RAT_PIT_I`, `ATC_RAT_PIT_D`
   - **Yaw**: `ATC_RAT_YAW_P`, `ATC_RAT_YAW_I`, `ATC_RAT_YAW_D`
   - **Altitude**: `PSC_POSZ_P`, `PSC_VELZ_P`, `PSC_ACCZ_P`
5. Введите рассчитанные значения
6. Нажмите "Write Params"

### Рекомендации по применению

1. **Начните с 70% от рассчитанных значений**
   ```
   P_actual = 0.7 × P_calculated
   I_actual = 0.7 × I_calculated
   D_actual = 0.7 × D_calculated
   ```

2. **Тестируйте постепенно:**
   - Начните с P коэффициента
   - Затем добавьте I
   - В конце настройте D

3. **Проведите тестовый полет:**
   - Летайте в режиме Stabilize
   - Делайте плавные маневры
   - Записывайте новые логи

4. **Итерируйте:**
   - Анализируйте новые логи
   - Повторяйте процесс при необходимости

## Примеры

### Пример 1: Анализ лога после первого полета
```bash
cargo run --release -- analyze \
  -i logs/first_flight.log \
  -a all \
  -m ziegler-nichols \
  -o results/first_tune.json
```

### Пример 2: Точная настройка только Roll оси
```bash
cargo run --release -- analyze \
  -i logs/tuning_flight.csv \
  -a roll \
  -m relay \
  -o results/roll_tune.json
```

### Пример 3: Экспорт данных для анализа в Excel/Python
```bash
cargo run --release -- extract \
  -i logs/flight.log \
  -o analysis/flight_data.csv
```

## Структура проекта

```
PIDSettings/
├── src/
│   ├── main.rs           # CLI интерфейс
│   ├── models.rs         # Структуры данных
│   ├── log_parser.rs     # Парсинг логов Ardupilot
│   └── pid_calculator.rs # Алгоритмы расчета PID
├── Cargo.toml            # Зависимости проекта
└── README.md             # Документация
```

## Зависимости

- **clap** - CLI парсинг
- **serde/serde_json** - Сериализация данных
- **csv** - Работа с CSV файлами
- **nalgebra** - Линейная алгебра
- **ndarray** - Массивы для численных расчетов
- **anyhow/thiserror** - Обработка ошибок

## Разработка

### Запуск тестов
```bash
cargo test
```

### Проверка кода
```bash
cargo clippy
```

### Форматирование
```bash
cargo fmt
```

## Дополнительные ресурсы

- [Ardupilot Documentation](https://ardupilot.org/copter/docs/tuning.html)
- [PID Tuning Guide](https://ardupilot.org/copter/docs/common-pid-tuning.html)
- [MAVLink Protocol](https://mavlink.io/)
- [Ziegler-Nichols Method](https://en.wikipedia.org/wiki/Ziegler%E2%80%93Nichols_method)

## Лицензия

MIT License - см. файл LICENSE

## Авторы

PID Tuner Team

## Поддержка

По вопросам и предложениям создавайте Issues в репозитории.
