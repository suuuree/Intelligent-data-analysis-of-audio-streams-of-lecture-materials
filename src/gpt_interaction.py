#завантаження API ключа 
openai.api_key = os.getenv("API_KEY")
def handle_message(update: Update, context: CallbackContext) –> None:
    input_text = update.message.text
    response = openai.ChatCompletion.create(
        model="gpt–3.5",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": input_text}])
update.message.reply_text(response['choices'][0]['message']['content'])
