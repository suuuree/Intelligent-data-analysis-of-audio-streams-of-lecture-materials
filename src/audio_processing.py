import librosa
import librosa.display
import IPython.display as ipd
import os
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment, effects
from pydub.silence import detect_nonsilent

ffmpeg_path = r"C:\Windows\System32\ffmpeg.exe"
ffprobe_path = r"C:\Windows\System32\ffprobe.exe"
AudioSegment.converter = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path

def convert_mp3_to_wav(input_mp3_path, original_wav_path):
    try:
        audio = AudioSegment.from_mp3(input_mp3_path)
        audio.export(original_wav_path, format="wav")
        print(f"Converted {input_mp3_path} to {original_wav_path}")
    except Exception as e:
        print(f"Error processing {input_mp3_path}: {str(e)}")

def clean_audio(original_wav_path, cleaned_wav_path):
    try:
        sound = AudioSegment.from_wav(original_wav_path)
        normalized_sound = effects.normalize(sound)
        nonsilent_ranges = detect_nonsilent(normalized_sound, min_silence_len=100, silence_thresh=normalized_sound.dBFS-14)
        nonsilent_sound = sum(normalized_sound[start:end] for start, end in nonsilent_ranges)
        low_pass_filtered = nonsilent_sound.low_pass_filter(3000)
        compressed_audio = low_pass_filtered.compress_dynamic_range()
        compressed_audio.export(cleaned_wav_path, format="wav")  
        print(f"Cleaned audio saved to {cleaned_wav_path}")
    except Exception as e:
        print(f"Error cleaning {original_wav_path}: {str(e)}")

# Шляхи до файлів
input_mp3_path = r'D:\zagruzki\DIPLOMA\dip\psch\Audio\audio\ex.mp3'
original_wav_path = r'D:\zagruzki\DIPLOMA\dip\psch\Audio\audio\original_output.wav'
cleaned_wav_path = r'D:\zagruzki\DIPLOMA\dip\psch\Audio\audio\ex_cleaned.wav'

# Проводимо конвертацію та очищення
convert_mp3_to_wav(input_mp3_path, original_wav_path)
clean_audio(original_wav_path, cleaned_wav_path)

# Завантаження і аналіз оригінального аудіо
x_orig, sr_orig = librosa.load(original_wav_path)
x_clean, sr_clean = librosa.load(cleaned_wav_path)

""" # Візуалізація оригінальної хвильової форми
plt.figure(figsize=(14, 5))
plt.plot(np.linspace(0, len(x_orig) / sr_orig, num=len(x_orig)), x_orig, label='Original')
plt.title('Оригінальна хвильова форма аудіофайлу')
plt.xlabel('Час (секунди)')
plt.ylabel('Амплітуда')
plt.legend()
plt.show() """

""" # Візуалізація очищеної хвильової форми
plt.figure(figsize=(14, 5))
plt.plot(np.linspace(0, len(x_clean) / sr_clean, num=len(x_clean)), x_clean, color='r', label='Cleaned')
plt.title('Очищена хвильова форма аудіофайлу')
plt.xlabel('Час (секунди)')
plt.ylabel('Амплітуда')
plt.legend()
plt.show() """

# Завантаження і аналіз очищеного аудіо
x, sr = librosa.load(cleaned_wav_path)
print(x.shape, sr)  # x.shape = (276480,), sr = 22050

""" # Будуємо візуалізацію в формі хвиль
plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, len(x) / sr, num=len(x)), x)
plt.title('Волнова форма аудіофайлу')
plt.xlabel('Час (секунди)')
plt.ylabel('Амплітуда')
plt.show()
 """
""" # Обробка спектограми
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X) + np.finfo(float).eps)  # Избегаем логарифма нуля
plt.figure(figsize=(10,5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(label='Гучність (dB)')
plt.title('Спектограма аудіофайлу')
plt.show() """

""" # Перцептивне зважування
freq = librosa.core.fft_frequencies(sr=sr) + np.finfo(float).eps
mag = librosa.perceptual_weighting(abs(X)**2 + np.finfo(float).eps, freq)  # Избегаем логарифма нуля
librosa.display.specshow(mag, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(label='Вага')
plt.title('Перцептивне зважування спектограми')
plt.show() """

# Автокореляція
r = librosa.autocorrelate(x, max_size=5000)
sample = r[:300]
plt.figure(figsize=(10,5))
plt.plot(sample)
plt.title('Автокореляція аудіосигналу')
plt.xlabel('Затримка')
plt.ylabel('Автокореляція')
plt.show()

# Хроматичні ознаки
sound_len = 400
chrom = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=sound_len)  # Указываем 'y='
plt.figure(figsize=(10,5))
librosa.display.specshow(chrom, x_axis='time', y_axis='chroma', hop_length=sound_len)
plt.colorbar(label='Інтенсивність')
plt.title('Хроматичні ознаки аудіосигналу')
plt.show()

# Мел-спектрограма
# Используем именованный аргумент 'y=' для передачи аудиосигнала в melspectrogram
# и проверим значение n_fft
S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=1024)
log = librosa.power_to_db(S, ref=np.max)
plt.figure(figsize=(10,5))
librosa.display.specshow(log,sr=sr,x_axis='time',y_axis='mel')
plt.colorbar(label='Гучність (dB)')
plt.title('Мел-спектрограма аудіофайлу')
plt.show()
