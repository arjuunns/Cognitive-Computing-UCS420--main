import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.probability import FreqDist
import re
import matplotlib.pyplot as plt
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

text = """Deadpool and Wolverine are finally uniting in Marvel's most anticipated team-up movie. Ryan Reynolds' signature humor clashes with Hugh Jackman's gruff intensity, creating perfect superhero chemistry. The R-rated adventure promises brutal action sequences and fourth-wall-breaking jokes that fans love. Set in the multiverse, the film explores new dimensions of both characters' legacies. With cameos from past X-Men films, it's a nostalgic celebration for longtime Marvel fans. This unconventional pairing might just redefine superhero team-ups forever."""

print("=== Q1: Text Processing ===")

# 1. Convert to lowercase and remove punctuation
clean_text = re.sub(r'[^\w\s]', '', text.lower())
print("\n1. Cleaned text:\n", clean_text)

# 2. Tokenize into words and sentences
words = word_tokenize(clean_text)
sentences = sent_tokenize(text)
print("\n2. Word tokens:", words[:10], "...")
print("   Sentence tokens:", sentences[:2])

# 3. Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]
print("\n3. After stopword removal:", filtered_words[:10], "...")

# 4. Word frequency distribution
fdist = FreqDist(filtered_words)
print("\n4. Most common words:")
fd=FreqDist(filtered_words)
print(fd)
fd.plot(10, title="Top Words")
# Q2: Stemming and Lemmatization
print("\n=== Q2: Stemming and Lemmatization ===")

# Initialize stemmers and lemmatizer
porter = PorterStemmer()
lancaster = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

# Apply to filtered words from Q1
stemmed_porter = [porter.stem(word) for word in filtered_words]
stemmed_lancaster = [lancaster.stem(word) for word in filtered_words]
lemmatized = [lemmatizer.lemmatize(word) for word in filtered_words]

print("\nOriginal words:", filtered_words[:10])
print("Porter stems:", stemmed_porter[:10])
print("Lancaster stems:", stemmed_lancaster[:10])
print("Lemmatized:", lemmatized[:10])

# Q3: Regular Expressions and Text Splitting
print("\n=== Q3: Regular Expressions and Text Splitting ===")

# 2a. Words with more than 5 letters
long_words = re.findall(r'\b\w{6,}\b', clean_text)
print("\na. Words with >5 letters:", long_words)

# 2b. Extract numbers (none in our text, but pattern would be)
numbers = re.findall(r'\d+', clean_text)
print("b. Numbers found:", numbers)

# 2c. Capitalized words (original text)
capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
print("c. Capitalized words:", capitalized)

# 3a. Split into alphabetic words
alpha_words = re.findall(r'\b[a-z]+\b', clean_text)
print("\na. Alphabetic words:", alpha_words[:10], "...")

# 3b. Words starting with vowels
vowel_words = re.findall(r'\b[aeiou][a-z]*\b', clean_text)
print("b. Words starting with vowels:", vowel_words)

# Q4: Custom Tokenization & Regex-based Text Cleaning
print("\n=== Q4: Custom Tokenization & Regex-based Text Cleaning ===")

def custom_tokenizer(text):
    # Keep contractions, hyphenated words, and decimal numbers
    pattern = r"""
        \b\w+(?:'\w+)?\b          # words with optional contractions
        | \b\d+\.\d+\b            # decimal numbers
        | \b\d+\b                 # integers
        | \b\w+(?:-\w+)+\b        # hyphenated words
    """
    return re.findall(pattern, text, re.VERBOSE)

# Sample text with additional elements for testing
test_text = text + " Contact me at esingla_be23@thapar.edu or visit https://eshansingla.com. Call +91 9876543210."

# Custom tokenization
tokens = custom_tokenizer(test_text.lower())
print("\n2. Custom tokens:", tokens)

# Regex substitutions
cleaned_text = test_text
cleaned_text = re.sub(r'\S+@\S+', '<EMAIL>', cleaned_text)  # emails
cleaned_text = re.sub(r'https?://\S+', '<URL>', cleaned_text)  # URLs
cleaned_text = re.sub(r'(\+\d{1,3} \d{10}|\d{3}-\d{3}-\d{4})', '<PHONE>', cleaned_text)  # phones

print("\n3. After regex substitutions:")
print(cleaned_text)