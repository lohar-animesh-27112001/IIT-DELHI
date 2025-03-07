ngrams = {
    "order": 3,
}

no_smoothing = {
    "method_name" : "NO_SMOOTH",
}

add_k = {
    "method_name" : "ADD_K",
    'k': 1.0,
}

stupid_backoff = {
    "method_name" : "STUPID_BACKOFF",
    'alpha': 0.4,
}

good_turing = {
    "method_name" : "GOOD_TURING",
}

interpolation = {
    "method_name" : "INTERPOLATION",
    'lambdas': [0.7, 0.3],
}

kneser_ney = {
    "method_name" : "KNESER_NEY",
    'discount': 0.75,
}

error_correction = {
    "internal_ngram_best_config" : {
        "method_name" : "NO_SMOOTH",
        'discount': 0.75,
    },
    "error_model": {
        "common_typos": {
            # 'teh': 'the',
            # 'adn': 'and',
            # 'recieve': 'receive',
        },
        "typo_probability": 0.9,
    },
}

additional_hyperparameters = {
    "rare_word_threshold": 5,
}
