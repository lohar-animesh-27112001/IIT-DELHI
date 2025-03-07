import numpy as np
import pandas as pd
from typing import List, Tuple
import re
from collections import defaultdict
# config.py

class NGramBase:
    def __init__(self, n : int = 2, lowercase: bool = True, remove_punctuation: bool = True):
        """
        Initialize basic n-gram configuration.
        :param n: The order of the n-gram (e.g., 2 for bigram, 3 for trigram).
        :param lowercase: Whether to convert text to lowercase.
        :param remove_punctuation: Whether to remove punctuation from text.
        """
        self.current_config = {"n": n, "lowercase": lowercase, "remove_punctuation": remove_punctuation, "method_name": "NGramBase"}

        # change code beyond this point
        self.n = n
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.total_counts = 0
        #

    def method_name(self) -> str:

        return f"Method Name: {self.current_config['method_name']}"

    def fit(self, data: List[List[str]]) -> None:
        """
        Fit the n-gram model to the data.
        :param data: The input data. Each sentence is a list of tokens.
        """
        for sentence in data:
            for i in range(len(sentence) - self.n + 1):
                self.ngram_counts[tuple(sentence[i:i + self.n])] += 1
                self.total_counts += 1
                self.context_counts[tuple(sentence[i:i + self.n - 1])] += 1
        # raise NotImplementedError

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text.
        :param text: The input text.
        :return: The list of tokens.
        """
        return re.split(r"[.?]", text)
        # raise NotImplementedError

    def prepare_data_for_fitting(self, data: List[str], use_fixed = False) -> List[List[str]]:
        """
        Prepare data for fitting.
        :param data: The input data.
        :return: The prepared data.
        """
        processed = []
        if not use_fixed:
            for text in data:
                processed.append(self.tokenize(self.preprocess(text)))
        else:
            for text in data:
                processed.append(self.fixed_tokenize(self.fixed_preprocess(text)))

        return processed

    def update_config(self, config) -> None:
        """
        Override the current configuration. You can use this method to update
        the config if required
        :param config: The new configuration.
        """
        self.current_config = config

    def preprocess(self, text: str) -> str:
        """
        Preprocess text before n-gram extraction.
        :param text: The input text.
        :return: The preprocessed text.
        """
        if(self.lowercase):
            return text.lower()
        return text
        # raise NotImplementedError

    def fixed_preprocess(self, text: str) -> str:
        """
        Removes punctuation and converts text to lowercase.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def fixed_tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the input text by splitting at spaces.
        """
        return text.split()

    def perplexity(self, text: str) -> float:
        """
        Compute the perplexity of the model given the text.
        :param text: The input text.
        :return: The perplexity of the model.
        """
        tokens = self.fixed_tokenize(self.fixed_preprocess(text))
        tokens = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        log_prob_sum = 0.0
        total_ngrams = len(tokens) - self.n + 1
        for i in range(total_ngrams):
            ngram = tuple(tokens[i:i + self.n])
            prob = self.probability(ngram)
            if prob > 0:
                log_prob_sum += np.log(prob)
            else:
                log_prob_sum += np.log(1e-10)
        average_log_prob = log_prob_sum / total_ngrams
        perplexity = np.exp(-average_log_prob)
        return perplexity
        # raise NotImplementedError
    
    def probability(self, ngram: Tuple[str, ...]) -> float:
        context = ngram[:-1]
        ngram_count = self.ngram_counts.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)
        if context_count == 0:
            return 0.0
        probability = ngram_count / context_count
        return probability

if __name__ == "__main__":
    tester_ngram = NGramBase(n=2)
    test_sentence = "This, is a ;test sentence. Where I want to go? now"
    train_sentence = "This is a test sentence. Where I want to go now"
    print(tester_ngram.method_name())
    sentences = tester_ngram.tokenize(train_sentence)
    print(sentences)
    tokenize_sentences = tester_ngram.prepare_data_for_fitting(sentences)
    print(tokenize_sentences)
    tester_ngram.fit(tokenize_sentences)
    print(tester_ngram.perplexity(test_sentence))
