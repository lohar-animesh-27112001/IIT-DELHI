#!/usr/bin/env python3
import os
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import transformers
from tqdm import tqdm
from datasets import Dataset
import random
import pandas as pd
from sacrebleu.metrics import BLEU
from transformers import (
    BartTokenizer, BartForConditionalGeneration,
    T5Tokenizer, T5ForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, 
    AutoTokenizer
)
from sklearn.model_selection import train_test_split
os.environ["WANDB_DISABLED"] = "true"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class GECConfig:
    """Configuration for the GEC model."""

    output_dir: str = "./gec_model_outputs"
    cache_dir: str = "./cache"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    entry_number = "2023EET2177"
    batch_correct_batch_size: int = 32 
    model_name = "facebook/bart-base"  # or "t5-base"
    batch_size = 4
    num_train_epochs = 1
    learning_rate = 5e-5
    max_source_length = 64
    max_target_length = 64
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.1

class M2Parser:
    """Parser for M2 formatted GEC data."""

    @staticmethod
    def parse_m2_file(filename: str) -> List[Dict]:
        """
        Parse an M2 file into a list of sentence dictionaries.

        Args:
            filename: Path to M2 file

        Returns:
            List of dictionaries with source and target sentences
        """
        data = []
        current_sentence = {}
        source_sentence = None
        corrections = []

        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if line.startswith('S '):
                    if source_sentence is not None and corrections:
                        current_sentence = {
                            'source': source_sentence,
                            'corrections': corrections
                        }
                        data.append(current_sentence)

                    source_sentence = line[2:]
                    corrections = []

                elif line.startswith('A '):
                    parts = line[2:].split("|||")
                    if len(parts) >= 3:
                        start_idx = int(parts[0].split()[0])
                        end_idx = int(parts[0].split()[1])
                        error_type = parts[1]
                        correction = parts[2]
                        corrections.append({
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'error_type': error_type,
                            'correction': correction
                        })

        if source_sentence is not None and corrections:
            current_sentence = {
                'source': source_sentence,
                'corrections': corrections
            }
            data.append(current_sentence)

        return data

    @staticmethod
    def apply_corrections(source: str, corrections: List[Dict]) -> str:
        """
        Apply corrections to a source sentence.

        Args:
            source: Source sentence
            corrections: List of correction dictionaries

        Returns:
            Corrected sentence
        """

        tokens = source.split()
        sorted_corrections = sorted(corrections, key=lambda x: x['start_idx'], reverse=True)

        for correction in sorted_corrections:
            start_idx = correction['start_idx']
            end_idx = correction['end_idx']
            corrected_text = correction['correction']

            if start_idx < len(tokens):
                del tokens[start_idx:end_idx]

                if corrected_text.strip():
                    corrected_tokens = corrected_text.split()
                    for i, token in enumerate(corrected_tokens):
                        tokens.insert(start_idx + i, token)

        corrected_sentence = ' '.join(tokens)

        return corrected_sentence

class GECorrector:
    """GEC system using the BART model."""

    def __init__(self, config: GECConfig):
        """
        Initialize the GEC system.

        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.device)
        tokenizer=None
        # Load tokenizer and model only if not provided
        if tokenizer is not None and model is not None:
            self.tokenizer = tokenizer
            self.model = model
        else:
            self.model = BartForConditionalGeneration.from_pretrained(config.model_name)

    def load_and_prepare_data(self, m2_file: str, val_ratio: float = 0.1, max_length: int = 128):
        """
        Load and preprocess training and validation data from M2 file.

        Args:
            m2_file: Path to M2 file.
            val_ratio: Ratio of validation split.
            max_length: Maximum token length for tokenizer.

        Returns:
            Tuple of tokenized train and validation HuggingFace Datasets.
        """
        logger.info(f"Parsing M2 file: {m2_file}")
        m2_data = M2Parser.parse_m2_file(m2_file)

        # Extract source and target pairs
        sources, targets = [], []
        for item in m2_data:
            src = item["source"]
            tgt = M2Parser.apply_corrections(src, item["corrections"])
            sources.append(src)
            targets.append(tgt)

        # Split into train and validation
        train_src, val_src, train_tgt, val_tgt = train_test_split(
            sources, targets, test_size=val_ratio, random_state=42
        )

        # Initialize tokenizer
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

        # Tokenize
        def tokenize_fn(example):
            model_inputs = self.tokenizer(
                example["source"],
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    example["target"],
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                )["input_ids"]
            model_inputs["labels"] = labels
            return model_inputs

        train_dataset = Dataset.from_dict({"source": train_src, "target": train_tgt}).map(tokenize_fn)
        val_dataset = Dataset.from_dict({"source": val_src, "target": val_tgt}).map(tokenize_fn)

        logger.info(f"Training examples: {len(train_dataset)} | Validation examples: {len(val_dataset)}")
        return train_dataset, val_dataset


    def train(self, train_dataset: Dataset, val_dataset: Dataset):
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """


        # create the train val split from train.m2
        
        logger.info("Starting training")
        # implement your code
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            save_strategy="epoch",
            logging_dir="./logs",
            predict_with_generate=True
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer
        )

        trainer.train()
        logger.info("Training completed")
        #raise NotImplementedError

        

    def batch_correct(self, sentences: List[str]) -> List[str]:
        """
        Correct grammatical errors in a batch of sentences.

        Args:
            sentences: List of sentences with potential grammatical errors

        Returns:
            List of corrected sentences
        """
        logger.info(f"Correcting {len(sentences)} sentences")
        self.model.eval()
        results = []
        for i in range(0, len(sentences), self.config.batch_correct_batch_size):
            batch = sentences[i:i + self.config.batch_correct_batch_size]
            inputs = [f"correct: {s}" if "t5" in self.config.model_name else s for s in batch]
            encodings = self.tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=self.config.max_source_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **encodings,
                    max_length=self.config.max_target_length,
                    num_beams=5,
                    length_penalty=1.0,
                    early_stopping=True
                )
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(decoded)
        return results
        #raise NotImplementedError

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.load(path, self.config)
        #raise NotImplementedError

    @classmethod
    def load(cls, model_path: str, config):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        #raise NotImplementedError
        
    def evaluate(self, source_sentences: List[str], reference_sentences: List[str]) -> float:
        """
        Evaluates the model using BLEU score.
        :param source_sentences: List of original uncorrected sentences.
        :param reference_sentences: List of reference corrected sentences.
        :return: BLEU score.
        """
        logger.info("Running evaluation using BLEU score...")
        predictions = self.batch_correct(source_sentences)

        # sacreBLEU expects a list of references as a list of list of strings
        bleu = BLEU()
        score = bleu.corpus_score(predictions, [reference_sentences])

        logger.info(f"BLEU score: {score.score:.2f}")
        return score.score

if __name__ == "__main__":
    import argparse

    """parser = argparse.ArgumentParser(description="GEC using BART")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--m2_file", type=str, help="Path to M2 file for training")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--source_file", type=str, help="Path to source sentences")
    parser.add_argument("--reference_file", type=str, help="Path to reference corrections")
    parser.add_argument("--correct", action="store_true", help="Correct sentences")
    parser.add_argument("--input_file", type=str, help="Path to input sentences")
    parser.add_argument("--output_file", type=str, help="Path to output file")
    parser.add_argument("--model_path", type=str, default="./gec_model_outputs", help="Path to save/load model")
    parser.add_argument("--test_m2_file", type=str, help="Path to M2 file for evaluation")
    args = parser.parse_args()"""

    config = GECConfig(output_dir='model_dir')

    # if args.train and args.m2_file:
    corrector = GECorrector(config)
    train_dataset, val_dataset = corrector.load_and_prepare_data('./train.m2')
    corrector.train(train_dataset, val_dataset)
    corrector.save('model_dir/trained_model')
    # else:/kaggle/input/data-set
    # corrector = GECorrector.load('model_dir/trained_model', config)
    df = pd.read_csv("./submission.csv")
    sentences = df["source"].tolist()

    # Step 3: Correct sentences
    corrected = corrector.batch_correct(sentences)
    
    # Step 4: Save to CSV
    df["prediction"] = corrected
    df.to_csv("./submission.csv", index=False)

    print("Corrected sentences saved to submission.csv")

    #if args.evaluate and args.source_file and args.reference_file:
        #results = corrector.evaluate('/kaggle/input/data-files/', '')
        #logger.info(f"Evaluation results: {results}")

    #if args.correct and args.input_file:
        #with open('', 'r', encoding='utf-8') as f:
         #   sentences = [line.strip() for line in f if line.strip()]

        #corrected_sentences = corrector.batch_correct(sentences)

        #if args.output_file:
         #   with open('', 'w', encoding='utf-8') as f:
          #      for sentence in corrected_sentences:
           #         f.write(f"{sentence}\n")
            #logger.info(f"Corrected sentences saved to ")
        #else:
         #   for original, corrected in zip(sentences, corrected_sentences):
          #      logger.info(f"Original: {original}")
           #     logger.info(f"Corrected: {corrected}")
            #    logger.info("-" * 50)  """