# PID Tuner - Python Version

**Автоматический расчет PID коэффициентов из логов Ardupilot**

Python версия программы с поддержкой:
- ✅ **.bin логи** с SD-карты (DataFlash формат)
- ✅ **.log логи** (текстовый формат)
- ✅ **.csv логи** (экспорт из Mission Planner)
- ✅ **Портативный .exe** без зависимостей

---

## 🚀 Быстрый старт (Портативная версия)

### Вариант 1: Готовый .exe файл (РЕКОМЕНДУЕТСЯ)

1. **Скачайте `pid_tuner.exe`** из релизов
2. **Положите в любую папку**
3. **Откройте командную строку** в этой папке
4. **Запустите:**
   ```cmd
   pid_tuner.exe analyze -i your_flight.bin
   ```

**Не нужен Python! Не нужен интернет! Просто скопируйте .exe на любой компьютер!**

---

## 📦 Сборка .exe (если у вас есть интернет)

### Windows:

1. **Откройте командную строку** в папке `python_version`
2. **Запустите:**
   ```cmd
   build_exe.bat
   ```
3. **Готово!** Файл `pid_tuner.exe` будет в папке `dist`

### Linux/Mac:

```bash
chmod +x build_exe.sh
./build_exe.sh
```

Файл `pid_tuner` будет в папке `dist`

---

## 💻 Запуск через Python (если .exe не работает)

### Установка зависимостей:

```bash
pip install -r requirements.txt
```

### Использование:

```bash
python pid_tuner.py analyze -i your_flight.bin
```

---

## 📖 Использование

### Анализ .bin лога с SD-карты

```bash
# Windows (.exe)
pid_tuner.exe analyze -i 00000001.bin

# Python
python pid_tuner.py analyze -i 00000001.bin
```

### Анализ конкретной оси

```bash
# Только Roll
pid_tuner.exe analyze -i flight.bin -a roll

# Только Pitch
pid_tuner.exe analyze -i flight.bin -a pitch

# Только Yaw
pid_tuner.exe analyze -i flight.bin -a yaw

# Только Altitude
pid_tuner.exe analyze -i flight.bin -a alt
```

### Сохранение результатов

```bash
# Сохранить в JSON и параметры Ardupilot
pid_tuner.exe analyze -i flight.bin -o results.json

# Будет создано:
# - results.json - результаты в JSON
# - results.params - параметры для Ardupilot
```

### Методы настройки

```bash
# Ziegler-Nichols (по умолчанию, для начинающих)
pid_tuner.exe analyze -i flight.bin -m ziegler-nichols

# Relay Method (более точный)
pid_tuner.exe analyze -i flight.bin -m relay

# System Identification (адаптивный)
pid_tuner.exe analyze -i flight.bin -m manual
```

### Экспорт данных в CSV

```bash
pid_tuner.exe extract -i flight.bin -o data.csv
```

---

## 📁 Поддерживаемые форматы

### 1. .bin файлы (ОСНОВНОЙ)
- Логи с SD-карты Pixhawk/Ardupilot
- Путь на SD: `APM/LOGS/00000XXX.BIN`
- Просто вставьте SD-карту и скопируйте файлы

### 2. .log файлы
- Текстовый формат Ardupilot
- Конвертируйте из .bin в Mission Planner

### 3. .csv файлы
- Экспорт из Mission Planner:
  1. Ctrl+F → "Log Browse"
  2. Откройте .bin лог
  3. Нажмите "Export to CSV"

---

## 🎯 Примеры использования

### Пример 1: Анализ полета квадрокоптера

```bash
pid_tuner.exe analyze -i copter_flight.bin -o copter_pids.json
```

Результат:
```
📈 Calculated PID Coefficients:
Method: ziegler-nichols
────────────────────────────────────────
Roll:     P: 0.0450, I: 0.0900, D: 0.0056
Pitch:    P: 0.0450, I: 0.0900, D: 0.0056
Yaw:      P: 0.1800, I: 0.0180, D: 0.0000
Altitude: P: 1.0000, I: 0.5000, D: 0.0000

💾 Results saved to: copter_pids.json
💾 Ardupilot parameters saved to: copter_pids.params

⚠️  ВАЖНО: Начните с 70% от рассчитанных значений!
```

### Пример 2: Тонкая настройка Roll оси

```bash
pid_tuner.exe analyze -i tuning_flight.bin -a roll -m relay -o roll_fine.json
```

### Пример 3: Анализ нескольких логов

```bash
for %%f in (*.bin) do (
    echo Analyzing %%f...
    pid_tuner.exe analyze -i %%f -o %%f.json
)
```

---

## 🔧 Применение результатов

### 1. Открыть файл .params

Программа создает файл `results.params`:
```
ATC_RAT_RLL_P 0.0450
ATC_RAT_RLL_I 0.0900
ATC_RAT_RLL_D 0.0056
ATC_RAT_PIT_P 0.0450
ATC_RAT_PIT_I 0.0900
ATC_RAT_PIT_D 0.0056
...
```

### 2. Умножить на 0.7 для безопасности

```
Roll P: 0.0450 × 0.7 = 0.0315
Roll I: 0.0900 × 0.7 = 0.0630
Roll D: 0.0056 × 0.7 = 0.0039
```

### 3. Применить в Mission Planner

1. Подключитесь к Pixhawk
2. Config → Full Parameter List
3. Найдите параметры (например, `ATC_RAT_RLL_P`)
4. Введите новые значения (× 0.7!)
5. Нажмите "Write Params"

### 4. Тестовый полет

- Летайте осторожно!
- Записывайте новый лог
- Повторите анализ при необходимости

---

## 📋 Требования для хорошего анализа

### Хороший лог содержит:
- ✅ 30-60 секунд полета
- ✅ Активные маневры (движения стиков)
- ✅ Плавное управление
- ✅ Режим Stabilize или AltHold

### Избегайте:
- ❌ Крашей и падений
- ❌ Статичного зависания
- ❌ Слишком короткие логи (<10 сек)
- ❌ Логи с ошибками датчиков

---

## ❓ Решение проблем

### Ошибка: "pymavlink is required"

**Если используете Python:**
```bash
pip install pymavlink
```

**Если используете .exe:**
- Это не должно происходить! Пересоберите .exe
- Или используйте готовый .exe из релизов

### Ошибка: "Insufficient data points"

- Лог слишком короткий
- Нужно минимум 10 секунд активного полета
- Убедитесь, что в логе есть маневры

### Ошибка: "Failed to parse log"

- Проверьте формат файла (.bin/.log/.csv)
- Убедитесь, что файл не поврежден
- Попробуйте конвертировать в .csv через Mission Planner

### .exe не запускается

- Проверьте Windows Defender / антивирус
- Попробуйте запустить от администратора
- Используйте Python версию напрямую

---

## 🏗️ Структура проекта

```
python_version/
├── pid_tuner.py       # Главный скрипт
├── models.py          # Модели данных
├── log_parser.py      # Парсер логов (.bin/.log/.csv)
├── pid_calculator.py  # Алгоритмы PID
├── requirements.txt   # Зависимости Python
├── build_exe.py       # Скрипт сборки .exe
├── build_exe.bat      # Windows сборка
├── build_exe.sh       # Linux/Mac сборка
└── README.md          # Эта документация
```

---

## 🔬 Алгоритмы

### 1. Ziegler-Nichols
- Классический метод
- Основан на анализе колебаний
- Подходит для большинства систем

### 2. Relay Method
- Метод Åström-Hägglund
- Более консервативный
- Лучше для нестабильных систем

### 3. System Identification
- Метод наименьших квадратов
- Адаптивная настройка
- Учитывает специфику полета

Подробности в файле `ALGORITHMS.md` в корне репозитория.

---

## 📝 Лицензия

MIT License

---

## 🤝 Поддержка

По вопросам создавайте Issues в репозитории GitHub.

---

## 🎓 Дополнительные ресурсы

- [Ardupilot PID Tuning Guide](https://ardupilot.org/copter/docs/tuning.html)
- [Mission Planner Documentation](https://ardupilot.org/planner/)
- [DataFlash Log Format](https://ardupilot.org/dev/docs/logmessages.html)
