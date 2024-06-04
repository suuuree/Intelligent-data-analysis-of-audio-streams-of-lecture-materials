import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

#налаштування TensorFlow для використання GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

#шляхи
train_data_path = 'D:/zagruzki/DIPLOMA/dip/psch/Audio/audio/processed/train_data.csv'
test_data_path = 'D:/zagruzki/DIPLOMA/dip/psch/Audio/audio/processed/test_data.csv'

#завантаження та підготовка даних
train_data = pd.read_csv(train_data_path, delimiter=',', encoding='utf-8')
test_data = pd.read_csv(test_data_path, delimiter=',', encoding='utf-8')

#огляд перших строк 
print("Train data head:")
print(train_data.head())

print("Test data head:")
print(test_data.head())

#перевірка існування необхідних стовпців
required_columns = ['path', 'text']
for column in required_columns:
    if column not in train_data.columns:
        raise KeyError(f"Column '{column}' not found in train data")
    if column not in test_data.columns:
        raise KeyError(f"Column '{column}' not found in test data")

#видалення пустих строк
train_data.dropna(subset=['text'], inplace=True)
test_data.dropna(subset=['text'], inplace=True)

#підрахунок унікальних речень після очистки
unique_sentences_count = train_data['text'].nunique()

#розділення даних
train_val_data, val_data = train_test_split(train_data, test_size=0.1, random_state=10)

#перевірка шляхів
def check_file_paths(paths):
    missing_files = []
    for path in paths:
        if not os.path.exists(path):
            missing_files.append(path)
            print(f"File not found: {path}")
    return missing_files

#функція для підготовки аудіоданих
def preprocess_audio(file_path, max_length=16000):
    audio_binary = tf.io.read_file(file_path)
    waveform, sample_rate = tf.audio.decode_wav(audio_binary, desired_channels=1)
    waveform = tf.squeeze(waveform, axis=-1) 

    #доповнення або обрезка до max_length
    waveform_length = tf.shape(waveform)[0]
    padding = tf.maximum(0, max_length - waveform_length)
    zero_padding = tf.zeros([padding], dtype=tf.float32)
    waveform = tf.concat([waveform, zero_padding], 0)
    waveform = waveform[:max_length]

    return tf.reshape(waveform, (max_length, 1))


def transcribe_audio(model, file_path, max_length=16000):
    audio = preprocess_audio(file_path, max_length)
    audio = tf.expand_dims(audio, axis=0) 
    predictions = model.predict(audio)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

def preprocess_audio_wrapper(file_path, label, max_length=16000):
    audio = tf.py_function(preprocess_audio, [file_path], tf.float32)
    audio.set_shape((max_length, 1))
    return audio, label

def create_dataset(file_paths, labels, max_length=16000):
    missing_files = check_file_paths(file_paths)
    if missing_files:
        raise FileNotFoundError(f"Missing files: {missing_files}")
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(lambda x, y: preprocess_audio_wrapper(x, y, max_length), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset

train_files = train_val_data['path'].tolist()
train_labels = train_val_data['text'].apply(lambda x: hash(x) % unique_sentences_count).tolist()

val_files = val_data['path'].tolist()
val_labels = val_data['text'].apply(lambda x: hash(x) % unique_sentences_count).tolist()

test_files = test_data['path'].tolist()
test_labels = test_data['text'].apply(lambda x: hash(x) % unique_sentences_count).tolist()

print("Checking train files...")
train_missing = check_file_paths(train_files)

print("Checking validation files...")
val_missing = check_file_paths(val_files)

print("Checking test files...")
test_missing = check_file_paths(test_files)

if train_missing:
    raise FileNotFoundError("Train files missing.")
if val_missing:
    raise FileNotFoundError("Validation files missing.")
if test_missing:
    raise FileNotFoundError("Test files missing.")

train_ds = create_dataset(train_files, train_labels)
val_ds = create_dataset(val_files, val_labels)
test_ds = create_dataset(test_files, test_labels)

def ASRModel(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(32, kernel_size=10, activation='relu')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, kernel_size=10, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, kernel_size=10, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


max_length = 16000  
input_shape = (max_length, 1)
num_classes = unique_sentences_count
model = ASRModel(input_shape, num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

try:
    model.fit(train_ds, validation_data=val_ds, epochs=10)
except Exception as e:
    print(f"Error during training: {e}")

try:
    test_loss, test_acc = model.evaluate(test_ds)
    print(f'Test Accuracy: {test_acc:.2f}')
except Exception as e:
    print(f"Error during evaluation: {e}")


audio_file_path = 'D:/zagruzki/DIPLOMA/dip/psch/Audio/audio/ex_cleaned.wav'

predicted_class = transcribe_audio(model, audio_file_path, max_length)

train_data = pd.read_csv('D:/zagruzki/DIPLOMA/dip/psch/Audio/audio/processed/train_data.csv', delimiter=',', encoding='utf-8')
unique_sentences_count = train_data['text'].nunique()

predicted_text = train_data.loc[train_data['text'].apply(lambda x: hash(x) % unique_sentences_count) == predicted_class, 'text'].values[0]
print(f'Розпізнаний текст: {predicted_text}')


# Обучение модели
#history = model.fit(train_ds, validation_data=val_ds, epochs=10)
"""""
# Визуализация истории обучения
#def plot_history(history):
#    plt.figure(figsize=(12, 4))

    # Потери
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Тренувальні втрати')
    plt.plot(history.history['val_loss'], label='Втрати при валідації')
    plt.xlabel('Епоха')
    plt.ylabel('Втрата')
    plt.title('Втрати на навчання та валідацію')
    plt.legend()

    # Точность
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Точність тренувань')
    plt.plot(history.history['val_accuracy'], label='Точність валідації')
    plt.xlabel('Епоха')
    plt.ylabel('Точність')
    plt.title('Точність навчання та валідації')
    plt.legend()

    plt.show()

plot_history(history)
"""""
"""""
# Функция для визуализации истории обучения
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Диаграмма потерь
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Втрата на тренувальних даних')
    plt.plot(history.history['val_loss'], label='Втрата на валідаційних даних')
    plt.xlabel('Епоха')
    plt.ylabel('Втрата')
    plt.title('Динаміка втрати під час навчання')
    plt.legend()

    # Графік точності
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Точність на тренувальних даних')
    plt.plot(history.history['val_accuracy'], label='Точність на валідаційних даних')
    plt.xlabel('Епоха')
    plt.ylabel('Точність')
    plt.title('Динаміка точності під час навчання')
    plt.legend()

    plt.tight_layout()
    plt.show()
"""""
"""""
# Функція для побудови ROC-кривої
def plot_multiclass_roc(model, test_ds, num_classes):
    y_true = []
    y_scores = []

    for x_batch, y_batch in test_ds:
        y_true.extend(y_batch.numpy())
        y_scores.extend(model.predict(x_batch))

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Бинаризация меток
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    # ROC и AUC для каждого класса
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        if np.sum(y_true_bin[:, i]) == 0:
            print(f"Клас {i} не має позитивних прикладів у тестовому наборі.")
            continue
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Построение ROC-кривых для каждого класса
    plt.figure(figsize=(10, 8))
    colors = plt.get_cmap('tab10', num_classes)
    
    for i in range(num_classes):
        if i not in fpr:
            continue
        plt.plot(fpr[i], tpr[i], color=colors(i),
                 lw=2, label=f'Клас {i} (площа = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Хибнопозитивна частота')
    plt.ylabel('Істиннопозитивна частота')
    plt.title('ROC-крива для багатокласової класифікації')
    plt.legend(loc='lower right')
    plt.show()

# Пример использования функций
history = model.fit(train_ds, validation_data=val_ds, epochs=10)
plot_training_history(history)

num_classes = unique_sentences_count  # Убедитесь, что num_classes правильно определено
plot_multiclass_roc(model, test_ds, num_classes)
"""""
