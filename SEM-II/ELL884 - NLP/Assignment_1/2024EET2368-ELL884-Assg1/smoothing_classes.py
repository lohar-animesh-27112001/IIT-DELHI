from ngram import NGramBase
from config import *
import numpy as np
import pandas as pd
from typing import List, Tuple
import re
from collections import defaultdict

class NoSmoothing(NGramBase):

    def __init__(self, n: int = 2, lowercase: bool = True, remove_punctuation: bool = True):

        super(NoSmoothing, self).__init__(n=n, lowercase = lowercase, remove_punctuation = remove_punctuation)
        self.update_config(no_smoothing)

    def probability(self, ngram: Tuple[str, ...]) -> float:
        if len(ngram) != self.n:
            raise ValueError(f"Expected {self.n}-gram, but got {len(ngram)}-gram.")
        context = ngram[:-1]
        if self.total_counts == 0:
            return 0.0
        context_count = self.context_counts.get(context,0)
        ngram_count = self.ngram_counts.get(ngram,0)
        if context_count == 0:
            return 0.0
        return ngram_count / context_count

class AddK(NGramBase):

    def __init__(self, n: int = 2, k: float = 1.0, lowercase: bool = True, remove_punctuation: bool = True):
        super(AddK, self).__init__(n, lowercase, remove_punctuation)
        self.k = k
        self.vocabulary_size = 0
        self.update_config(add_k)

    def fit(self, data: List[List[str]]) -> None:
        super().fit(data)
        vocabulary = set()
        for sentence in data:
            for word in sentence:
                vocabulary.add(word)
        self.vocabulary_size = len(vocabulary)
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        for sentence in data:
            for i in range(len(sentence) - self.n + 1):
                ngram = tuple(sentence[i:i + self.n])
                self.ngram_counts[ngram] += 1
                context = tuple(sentence[i:i + self.n - 1])
                self.context_counts[context] += 1

    def probability(self, ngram: Tuple[str, ...]) -> float:
        if len(ngram) != self.n:
            raise ValueError(f"Expected {self.n}-gram, but got {len(ngram)}-gram.")
        context = ngram[:-1]
        context_count = self.context_counts.get(context,0)
        ngram_count = self.ngram_counts.get(ngram,0)
        return (ngram_count + self.k) / (context_count + self.k * self.vocabulary_size)

    def perplexity(self, text: str) -> float:
        return super().perplexity(text)

class StupidBackoff(NGramBase):

    def __init__(self, n: int = 2, lowercase: bool = True, remove_punctuation: bool = True, alpha: float = 0.4):
        super(StupidBackoff, self).__init__(n, lowercase, remove_punctuation)
        self.alpha = alpha
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.unigram_counts = defaultdict(int)
        self.update_config(stupid_backoff)

    def fit(self, data: List[List[str]]) -> None:
        super().fit(data)
        for sentence in data:
            for word in sentence:
                self.unigram_counts[word] += 1
        for sentence in data:
            for i in range(len(sentence) - self.n + 1):
                ngram = tuple(sentence[i:i + self.n])
                self.ngram_counts[ngram] += 1
                context = tuple(sentence[i:i + self.n - 1])
                self.context_counts[context] += 1

    def probability(self, ngram: tuple) -> float:
        if not ngram:
            return 1.0
        ngram_count = self.ngram_counts.get(ngram, 0)
        context = ngram[:-1]
        context_count = self.context_counts.get(context, 0)
        if context_count > 0:
            return ngram_count / context_count
        else:
            shorter_ngram = ngram[1:]
            return self.alpha * self.probability(shorter_ngram)

    def perplexity(self, text: str) -> float:
        return super().perplexity(text)

class GoodTuring(NGramBase):

    def __init__(self, n: int = 2, lowercase: bool = True, remove_punctuation: bool = True):
        super(GoodTuring, self).__init__(n, lowercase, remove_punctuation)
        self.ngram_count_distribution = defaultdict(int)
        self.update_config(good_turing)

    def fit(self, data: List[List[str]]) -> None:
        super().fit(data)
        self.ngram_count_distribution = defaultdict(int)
        for count in self.ngram_counts.values():
            self.ngram_count_distribution[count] += 1
        self.total_ngrams = sum(self.ngram_count_distribution.values())

    def probability(self, ngram: Tuple[str, ...]) -> float:
        count = self.ngram_counts[ngram]
        N = self.total_ngrams
        Nc = self.ngram_count_distribution[count]
        Nc1 = self.ngram_count_distribution[count + 1]
        if N == 0:
            return 0.0
        if Nc == 0:
            if Nc1 == 0:
                return 0.0
            else:
                c_star = (count + 1) * (Nc1 / Nc)
        else:
            c_star = (count + 1) * (Nc1 / Nc)
        return c_star / N

    def perplexity(self, text: str) -> float:
        return super().perplexity(text)

class Interpolation(NGramBase):

    def __init__(self, n: int = 2, lambdas: Tuple[float] = (0.5, 0.5), lowercase: bool = True, remove_punctuation: bool = True):
        super(Interpolation, self).__init__(n, lowercase, remove_punctuation)
        self.lambdas = lambdas
        self.n = n
        if len(self.lambdas) < self.n:
            raise ValueError("Number of lambdas should be equal to n-gram order.")
        self.update_config(interpolation)

    def fit(self, data: List[List[str]]) -> None:
        super().fit(data)

    def probability(self, ngram: Tuple[str, ...]) -> float:
        prob = 0.0
        for i in range(self.n):
            n_order = i + 1
            lambda_val = self.lambdas[i]
            if len(ngram) >= n_order:
                sub_ngram = ngram[-n_order:]
                if n_order > 1:
                    context = sub_ngram[:-1]
                    ngram_count = self.ngram_counts.get(sub_ngram, 0)
                    context_count = self.context_counts.get(context, 0)
                    
                    if context_count > 0:
                        prob += lambda_val * (ngram_count / context_count)
                    else:
                        prob += lambda_val * (1 / len(self.ngram_counts))
                else:
                    unigram = sub_ngram[0]
                    unigram_count = self.ngram_counts.get((unigram,), 0)
                    total_unigrams = sum(self.ngram_counts.values())
                    
                    if total_unigrams > 0:
                        prob += lambda_val * (unigram_count / total_unigrams)
                    else:
                        prob += lambda_val * (1 / len(self.ngram_counts))
            else:
                lambda_val = 0.0
        return prob

    def perplexity(self, text: str) -> float:
        return super().perplexity(text)

class KneserNey(NGramBase):


    def __init__(self, n: int = 2, discount: float = 0.75, lowercase: bool = True, remove_punctuation: bool = True):
        super(KneserNey, self).__init__(n, lowercase, remove_punctuation)
        self.update_config(kneser_ney)
        self.discount = discount
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.unigram_counts = defaultdict(int)
        self.continuation_counts = defaultdict(set)
        self.vocabulary_size = 0
        
    def fit(self, data: List[List[str]]) -> None:
        super().fit(data)
        for sentence in data:
            for word in sentence:
                self.unigram_counts[word] += 1
        for sentence in data:
            for i in range(len(sentence) - self.n + 1):
                ngram = tuple(sentence[i:i + self.n])
                self.ngram_counts[ngram] += 1
                context = tuple(sentence[i:i + self.n - 1])
                self.context_counts[context] += 1
                if len(ngram) > 1:
                    word = ngram[-1]
                    self.continuation_counts[word].add(context)

    def probability(self, ngram: Tuple[str, ...]) -> float:
        if not ngram:
            return 0.0
        if len(ngram) == 1:
            word = ngram[0]
            numerator = len(self.continuation_counts[word])
            denominator = sum(len(contexts) for contexts in self.continuation_counts.values())
            if denominator == 0:
                return 0.0
            return numerator / denominator
        else:
            context = ngram[:-1]
            word = ngram[-1]
            discounted_count = max(self.ngram_counts[ngram] - self.discount, 0)
            if self.context_counts[context] > 0:
                ml_estimate = discounted_count / self.context_counts[context]
            else:
                ml_estimate = 0.0
            lambda_val = (self.discount / self.context_counts[context]) * len(self.continuation_counts[word]) \
                         if self.context_counts[context] > 0 else 0.0
            lower_order_ngram = (word,)
            lower_order_prob = self.probability(lower_order_ngram)
            return ml_estimate + lambda_val * lower_order_prob

    def perplexity(self, text: str) -> float:
        return super().perplexity(text)

if __name__ == "__main__":
    ns = NoSmoothing()
    print(ns.method_name())
    ak = AddK()
    print(ak.method_name())
    sbf = StupidBackoff()
    print(sbf.method_name())
    gt = GoodTuring()
    print(gt.method_name())
    ip = Interpolation()
    print(ip.method_name())
    kn = KneserNey()
    print(kn.method_name())

