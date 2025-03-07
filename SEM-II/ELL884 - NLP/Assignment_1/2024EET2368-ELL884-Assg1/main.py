from ngram import NGramBase
from config import *
from smoothing_classes import *
from error_correction import *
import numpy as np
import pandas as pd
from typing import List, Tuple
import re
from collections import defaultdict
import os

def load_data(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def load_misspelling_data(file_path: str) -> List[Tuple[str, str]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped_line = line.strip()
            if "&&" in stripped_line:
                correct, incorrect = stripped_line.split("&&", 1)
                data.append((correct.strip(), incorrect.strip()))
    return data

def main():
    data_dir = "./data"
    train1_path = os.path.join(data_dir, "train1.txt")
    train2_path = os.path.join(data_dir, "train2.txt")
    train1_data = load_data(train1_path)
    train2_data = load_data(train2_path)
    train_data = train1_data + train2_data
    corrector = SpellingCorrector()
    print("Training the spelling corrector...")
    corrector.fit(train_data)
    print("Spelling corrector trained.")
    misspelling_path = os.path.join(data_dir, "misspelling_public.txt")
    misspelling_data = load_misspelling_data(misspelling_path)
    print("Evaluating the spelling corrector...\n")
    correct_count = 0
    total_count = len(misspelling_data)
    output_lines = []
    for correct_text, incorrect_text in misspelling_data:
        incorrect_tokens = corrector.internal_ngram.fixed_tokenize(corrector.internal_ngram.fixed_preprocess(incorrect_text))
        corrected_tokens = corrector.correct(incorrect_tokens)
        corrected_text = " ".join(corrected_tokens)
        if corrected_text == correct_text:
            correct_count += 1
        else:
            output_line = (f"Incorrect Correction: <{correct_text}> && <{corrected_text}> \n")
            output_lines.append(output_line)
            print(output_line)
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    with open("./output/output.txt", "w") as f:
        for line in output_lines:
            f.write(line + "\n")

if __name__ == "__main__":
    main()