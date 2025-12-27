from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
import re
import string

tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
lemmatizer = WordNetLemmatizer()
stopwords_english = set(stopwords.words('english')) - {'up', 'down', 'from', 'over', 'under'}  # keep finance-relevant words

ACRONYMS = {"eps", "q4", "q3", "q2", "q1", "ebitda", "fwd", "p/e", "p/s", "p/b", "p/cf", "cf", "net", "rev", "netrev", "netrevqoq", "netrevyoy",}


def process_text_to_words(text: str) -> list:
    """Advanced financial text processor."""

    # Remove hyperlinks
    text = re.sub(r'https?://[^\s\n\r]+', '', text)
    # Remove hashtag symbol but keep the word
    text = re.sub(r'#', '', text)

    # Tokenize
    tokens = tokenizer.tokenize(text)
    cleaned_tokens = []

    for word in tokens:
        word_lower = word.lower()

        # Preserve tickers like $AAPL
        if word_lower.startswith('$') and len(word_lower) > 1:
            cleaned_tokens.append(word_lower)
            continue

        # Remove words that are mostly numbers (e.g., 123, 0.63), unless in ACRONYMS
        if any(char.isdigit() for char in word_lower) and word_lower not in ACRONYMS:
            continue

        # Remove stopwords and pure punctuation
        if word_lower not in stopwords_english and word_lower not in string.punctuation:
            lemma = lemmatizer.lemmatize(word_lower)
            cleaned_tokens.append(lemma)

    return cleaned_tokens
