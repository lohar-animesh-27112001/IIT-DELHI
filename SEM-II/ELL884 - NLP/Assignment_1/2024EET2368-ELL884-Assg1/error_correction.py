from smoothing_classes import *
from config import error_correction
import numpy as np
import pandas as pd

class SpellingCorrector:

    def __init__(self):

        self.correction_config = error_correction
        self.internal_ngram_name = self.correction_config['internal_ngram_best_config']['method_name']

        if self.internal_ngram_name == "NO_SMOOTH":
            self.internal_ngram = NoSmoothing()
        elif self.internal_ngram_name == "ADD_K":
            self.internal_ngram = AddK()
        elif self.internal_ngram_name == "STUPID_BACKOFF":
            self.internal_ngram = StupidBackoff()
        elif self.internal_ngram_name == "GOOD_TURING":
            self.internal_ngram = GoodTuring()
        elif self.internal_ngram_name == "INTERPOLATION":
            self.internal_ngram = Interpolation()
        elif self.internal_ngram_name == "KNESER_NEY":
            self.internal_ngram = KneserNey()

        self.internal_ngram.update_config(self.correction_config['internal_ngram_best_config'])
        
        self.internal_ngram.n = ngrams['order']
        self.word_probabilities = defaultdict(float)
        self.error_probabilities = defaultdict(lambda: defaultdict(float))

    def fit(self, data: List[str]) -> None:
        """
        Fit the spelling corrector model to the data.
        :param data: The input data.
        """
        processed_data = self.internal_ngram.prepare_data_for_fitting(data, use_fixed=True)
        self.internal_ngram.fit(processed_data)
        for word in data:
            self.word_probabilities[word] += 1
        total_count = sum(self.word_probabilities.values())
        for word in self.word_probabilities:
            self.word_probabilities[word] /= total_count
        common_typos = self.correction_config['error_model']['common_typos']
        for typo, correct in common_typos.items():
            self.error_probabilities[typo][correct] = self.correction_config['error_model']['typo_probability']
            self.error_probabilities[typo][typo] = 1 - self.correction_config['error_model']['typo_probability']

    def correct(self, text: List[str]) -> List[str]:
        """
        Correct the input text.
        :param text: The input text.
        :return: The corrected text.
        """
        ## there will be an assertion to check if the output text is of the same
        ## length as the input text
        corrected_text = []
        for word in text:
            candidates = list(self.error_probabilities[word].keys()) if word in self.error_probabilities else []
            if not candidates:
                candidates = [word]
            if not candidates or word in self.word_probabilities:
                corrected_text.append(word)
            else:
                candidate_scores = {}
                for candidate in candidates:
                    error_prob = self.error_probabilities[word][candidate] if candidate in self.error_probabilities[word] else 0.0001
                    context = tuple(corrected_text[-(self.internal_ngram.n - 1):])
                    while len(context) < (self.internal_ngram.n - 1):
                        context = ("<s>",) + context
                    context = context[-(self.internal_ngram.n - 1):]
                    ngram = context + (candidate,)
                    if len(ngram) != self.internal_ngram.n:
                        continue
                    language_model_prob = self.internal_ngram.probability(ngram)
                    candidate_scores[candidate] = error_prob * language_model_prob
                if candidate_scores:
                    best_candidate = max(candidate_scores, key=candidate_scores.get)
                    corrected_text.append(best_candidate)
                else:
                    corrected_text.append(word)
        text = corrected_text
        return text
