# File: src/data_processing/preprocessor.py
"""Module for raw text preprocessing."""
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect, LangDetectException
import emoji

# Download necessary NLTK resources
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")


class TextCleaner:
    """Handles text cleaning operations."""

    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")  # URLs
    SPECIAL_CHAR_PATTERN = re.compile(r"[^a-zA-Z\s]")  # Non-alphabetic characters
    WHITESPACE_PATTERN = re.compile(r"\s+")  # Extra whitespace
    PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")  # Punctuation

    @classmethod
    def remove_urls(cls, text: str) -> str:
        """Removes URLs from the text."""
        return cls.URL_PATTERN.sub("", text)

    @staticmethod
    def remove_emojis(text: str) -> str:
        """Removes emojis from the text."""
        return emoji.replace_emoji(text, replace="")

    @classmethod
    def remove_punctuation(cls, text: str) -> str:
        """Removes punctuation from the text."""
        return cls.PUNCTUATION_PATTERN.sub("", text)

    @classmethod
    def remove_special_chars(cls, text: str) -> str:
        """Removes non-alphabetic characters."""
        return cls.SPECIAL_CHAR_PATTERN.sub("", text)

    @classmethod
    def normalize_whitespace(cls, text: str) -> str:
        """Normalizes extra whitespace."""
        return cls.WHITESPACE_PATTERN.sub(" ", text)


class TextPreprocessor:
    """Handles text preprocessing."""

    CONTRACTIONS = {
        "there's": "there is", "it's": "it is", "wasn't": "was not",
        "weren't": "were not", "can't": "cannot", "won't": "will not",
        "doesn't": "does not", "didn't": "did not", "don't": "do not",
        "I've": "I have", "we've": "we have", "you're": "you are",
        "he's": "he is", "she's": "she is", "we're": "we are",
        "they're": "they are", "who's": "who is", "what's": "what is",
        "that's": "that is", "how's": "how is",
        "u.s.": "united states", "govt": "government", "CFO": "chief financial officer",
        "IPO": "initial public offering", "AI": "artificial intelligence",
        "CEO": "chief executive officer", "NASDAQ": "nasdaq stock market",
        "NYT": "new york times", "SEC": "securities and exchange commission",
    }

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.cleaner = TextCleaner()

    def preprocess(self, text: str) -> str:
        """Cleans and normalizes text."""
        text = self.cleaner.remove_urls(text)
        text = self.cleaner.remove_emojis(text)
        text = self.cleaner.remove_punctuation(text)
        text = self._expand_contractions(text)
        text = self.cleaner.remove_special_chars(text)
        text = text.lower()
        text = self.cleaner.normalize_whitespace(text)
        text = self._lemmatize_and_remove_stopwords(text)
        return text.strip()

    def _expand_contractions(self, text: str) -> str:
        """Expands contractions and abbreviations."""
        for contraction, expansion in self.CONTRACTIONS.items():
            text = re.sub(rf"\b{re.escape(contraction)}\b", expansion, text, flags=re.IGNORECASE)
        return text

    def _lemmatize_and_remove_stopwords(self, text: str) -> str:
        """Lemmatizes words and removes stopwords."""
        words = word_tokenize(text)
        return " ".join([self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words])


class FinancialNewsProcessor:
    """Processes financial news dataset."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.dataframe = None
        self.preprocessor = TextPreprocessor()

    def execute_pipeline(self) -> pd.DataFrame:
        """Executes full text processing pipeline."""
        self._load_data()
        self._filter_english_content()
        self._sanitize_text()
        self._process_dates()
        self._remove_duplicates()
        return self.dataframe

    def _load_data(self) -> None:
        """Loads CSV into a DataFrame."""
        self.dataframe = pd.read_csv(self.data_path)

    def _filter_english_content(self) -> None:
        """Removes non-English news using langdetect."""
        self.dataframe = self.dataframe[self.dataframe["news"].apply(
            lambda text: self._is_english(text)
        )]

    def _is_english(self, text: str) -> bool:
        """Detects if the text is in English using langdetect."""
        try:
            return detect(text) == "en"
        except LangDetectException:
            return False

    def _sanitize_text(self) -> None:
        """Cleans and preprocesses the 'news' column."""
        self.dataframe["news"] = self.dataframe["news"].map(self.preprocessor.preprocess)

    def _process_dates(self) -> None:
        """Converts date column to datetime format."""
        self.dataframe["date"] = pd.to_datetime(self.dataframe["date"], errors="coerce")

    def _remove_duplicates(self) -> None:
        """Removes duplicate news entries."""
        self.dataframe.drop_duplicates(subset=["news"], keep="first", inplace=True)

    def save_processed_data(self, output_path: str) -> None:
        """Saves processed data to the processed folder."""
        self.dataframe.to_csv(output_path, index=False)


if __name__ == "__main__":
    # Define the raw data path and processed data path
    raw_data_path = "data/raw/news.csv"
    processed_data_path = "data/processed/cleaned_news.csv"

    # Initialize the FinancialNewsProcessor
    processor = FinancialNewsProcessor(raw_data_path)

    # Execute the pipeline to process data
    processed_data = processor.execute_pipeline()

    # Save the processed data to the specified output path
    processor.save_processed_data(processed_data_path)

    print(f"Data preprocessing completed and saved to {processed_data_path}")