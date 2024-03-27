# LectureMate
LectureMate is an innovative tool designed to assist in the educational process that utilizes advanced machine learning and natural language processing (NLP) technologies to convert audio lectures into text format. 

## Main Functions
- **Audio to Text Conversion**: Utilizes advanced speech recognition algorithms to convert lecture audio into text with high accuracy.
- **Audio Data Processing**: Pre-process audio to improve recognition quality, including noise removal and volume normalization.
- **Text Analysis**: Identify key lecture elements and enrich the text by analyzing and highlighting main ideas.
- **Interactivity**: Interact with the text through integration with ChatGPT API to provide answers to questions and additional information on lecture topics.
- **Comfortable User Interface**: Integration with Telegram bot to simplify student and faculty access and use of the system.

## Project strucutre
**data/:** Эта директория служит для хранения всех данных, связанных с проектом. Она включает в себя поддиректории:
**audio/:** Здесь находятся исходные аудиофайлы с лекциями, которые будут использоваться для распознавания и анализа.
**processed/:** Сюда сохраняются все обработанные данные, например, текстовые версии аудиофайлов или результаты анализа.
**models/:** Папка предназначена для хранения обученных моделей машинного обучения или нейронных сетей.
**notebooks/:** В этой директории располагаются Jupyter ноутбуки, которые можно использовать для экспериментов, исследования данных и прототипирования алгоритмов.
**src/:** Основная папка с исходным кодом проекта. Она включает в себя модули:
**__init__.py:** Файл, делающий директорию Python пакетом, что позволяет использовать модули в других частях проекта.
**audio_processing.py:** Модуль, содержащий функции и классы для обработки аудиофайлов (например, удаление шумов, нормализация громкости).
**speech_recognition.py:** Здесь реализованы функции для распознавания речи и преобразования аудио в текст.
**gpt_interaction.py:** Модуль для взаимодействия с API ChatGPT или аналогичными сервисами для анализа текста и получения ответов на вопросы.
**data_preprocessing.py:** Содержит функции для предварительной обработки данных, например, очистка текста, токенизация.
**model_training.py:** Модуль для обучения моделей машинного обучения или нейронных сетей на обработанных данных.
**requirements.txt:** Файл с перечислением всех необходимых библиотек и их версий для установки через pip install -r requirements.txt.

**README.md:** Файл Markdown с описанием проекта, его функциональности, инструкциями по установке и использованию.

.gitignore: Файл для указания git, какие файлы или директории игнорировать (например, логи, временные файлы или личные конфигурационные файлы).
