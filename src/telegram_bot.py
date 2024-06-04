from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import audio_processing  #модуль для конвертації аудіо
import data_preprocessing  #модуль для препроцесингу аудіо
import AsrModel from model_training  #модель для розпізнавання мови

def start(update: Update, context: CallbackContext) –> None:
    update.message.reply_text(Привіт! Я Telegram–бот, який допомагає студентам в оброці аудіо–лекцій, в меню Ви можете подивитись команди. Я вже готовий обробити ваш файл. Надішліть його командою /file або просто надішліть мені його.')
def handle_audio(update: Update, context: CallbackContext) –> None:
    file = context.bot.getFile(update.message.audio.file_id)
    file.download('audio.ogg')
    #конвертація і очищення аудіо
    audio_path = audio_processing.convert_and_clean_audio('audio.ogg')
#транскрибування
transcribed_text = AsrModel(audio_path)
    #обробка тексту
    processed_text = data_preprocessing.process_text(text)
    #відправка обробленого тексту користувачу
    update.message.reply_text(f'Оброблений текст: {processed_text}')
def main():
    application = Application.builder().token('TOKEN').build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.AUDIO, handle_audio))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling()
if __name__ == '__main__':
    main()
