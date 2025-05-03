import os
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from wordcloud import WordCloud
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
nltk.download('all')

text_q1 = "Marvel superheroes like Iron Man, Spider-Man, and Thor are incredible. They inspire us with their bravery and strength. Marvel movies combine action, humor, and emotion in perfect harmony. These stories remind us to be selfless and courageous. Watching Avengers assemble is always thrilling, and their adventures keep us on the edge of our seats."

text_lower = text_q1.lower()
text_cleaned = re.sub(r'[^\w\s]', '', text_lower)

sentences = sent_tokenize(text_cleaned)
words_nltk = word_tokenize(text_cleaned)
words_split = text_cleaned.split()

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words_nltk if word not in stop_words]

fdist = nltk.FreqDist(filtered_words)

words_only = re.findall(r'\b[a-zA-Z]+\b', text_cleaned)
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

texts_q3 = [
    "Iron Man's suit is made of advanced technology.",
    "Spider-Man's agility and strength are unmatched.",
    "Thor wields Mjolnir, a magical hammer."
]
cv = CountVectorizer()
bow_matrix = cv.fit_transform(texts_q3)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(texts_q3)
keywords = [tfidf.get_feature_names_out()[idx] for idx in tfidf_matrix.toarray().argsort(axis=1)[:, -3:]]

text1 = "Iron Man and Spider-Man are iconic heroes."
text2 = "Spider-Man's powers and Iron Man's suit make them strong."
words1 = set(word_tokenize(re.sub(r'[^\w\s]', '', text1.lower())))
words2 = set(word_tokenize(re.sub(r'[^\w\s]', '', text2.lower())))
jaccard_similarity = len(words1 & words2) / len(words1 | words2)

texts_tfidf = TfidfVectorizer()
vectors = texts_tfidf.fit_transform([text1, text2])
cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])

review = "The Avengers movie was absolutely incredible."
blob = TextBlob(review)
polarity = blob.sentiment.polarity
subjectivity = blob.sentiment.subjectivity
wc = WordCloud().generate(review)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

text_train = "Iron Man is a genius who created a suit to save the world."
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_train])
sequences = []
words = text_train.split()

for i in range(1, len(words)):
    seq = words[:i + 1]
    tokenized_seq = tokenizer.texts_to_sequences([' '.join(seq)])[0]
    sequences.append(tokenized_seq)

padded = pad_sequences(sequences)

model = Sequential()
model.add(Embedding(input_dim=50, output_dim=10, input_length=padded.shape[1]))
model.add(LSTM(50))
model.add(Dense(50, activation='relu'))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.summary()