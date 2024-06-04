import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from spacy.lang.ru import Russian
from spacy.lang.uk import Ukrainian

#завантаження моделей spaCy
nlp_ru = spacy.load('ru_core_news_sm')
nlp_uk = Ukrainian()  # Завантаження української моделі

#стоп-слова для української мови
ukrainian_stopwords = [
    "а", "але", "б", "без", "більше", "був", "була", "були", "було", "буде",
    "буду", "будемо", "будете", "будуть", "в", "вам", "вами", "вас", "ваш", "ваша",
    "ваше", "ваші", "вже", "ви", "від", "він", "вона", "вони", "воно", "все",
    "всі", "всім", "всіх", "відомо", "га", "два", "для", "до", "его", "є",
    "ж", "з", "за", "і", "й", "коли", "ком", "кого", "кому", "мене", "мені",
    "ми", "мною", "мого", "на", "нам", "нами", "наш", "наша", "наше", "наші",
    "не", "него", "неї", "немає", "ні", "ним", "ними", "ніж", "ній", "ніяк", 
    "ну", "ось", "під", "при", "про", "та", "так", "така", "таке", "такі", 
    "також", "там", "тебе", "тебен", "тех", "ти", "то", "тобі", "того", "тоді", 
    "той", "только", "том", "тому", "тут", "ті", "тільки", "у", "усі", "усе", 
    "хоча", "чого", "чому", "через", "це", "цей", "цим", "цими", "цього", "цьому", 
    "час", "як", "яка", "який", "яких", "якої", "її", "їм", "їх", "їхній", "їхні", 
    "якщо"
]
print(ukrainian_stopwords)

def preprocess_text(text, lang='ru'):
    if lang == 'ru':
        doc = nlp_ru(text)
        stopwords_list = russian_stopwords
    elif lang == 'uk':
        doc = nlp_uk(text)
        stopwords_list = ukrainian_stopwords
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.text.lower() not in stopwords_list]
    return " ".join(tokens)

def extract_entities(text, lang='ru'):
    nlp = nlp_ru if lang == 'ru' else nlp_uk
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_nouns(text, lang='ru'):
    nlp = nlp_ru if lang == 'ru' else nlp_uk
    doc = nlp(text)
    nouns = [token.text for token in doc if token.pos_ == 'NOUN']
    return nouns

def summarize_text(text, lang='ru'):
    nlp = nlp_ru if lang == 'ru' else nlp_uk
    doc = nlp(text)
    sentences = list(doc.sents)
    return " ".join(str(sentences[i]) for i in range(min(3, len(sentences))))

#демонстраційний виклик
text = "Введіть сюди ваш текст для обробки."
processed_text = preprocess_text(text, lang='ru')
entities = extract_entities(processed_text, lang='ru')
nouns = extract_nouns(processed_text, lang='ru')
summary = summarize_text(processed_text, lang='ru')

print("Оброблений текст:", processed_text)
print("Сутності:", entities)
print("Іменники:", nouns)
print("Резюме:", summary)
