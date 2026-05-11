import pytest
import polars as pl
import numpy as np
from bio_spread_reborn.data.loader import DataPipeline
from bio_spread_reborn.data.tokenizer import GeneTokenizer

def test_time_split():
    # Synthetic records
    df = pl.DataFrame({
        'backbone_id': ['B1', 'B2', 'B3', 'B4'],
        'year': [2018, 2019, 2020, 2021],
        'spread_label': [0, 1, 0, 1]
    })
    
    config = {
        'data': {
            'records_path': 'dummy.tsv',
            'split_year': 2020
        }
    }
    
    pipeline = DataPipeline(config)
    # Mocking read_csv and Path.exists to avoid file IO
    import unittest.mock as mock
    with mock.patch('polars.read_csv', return_value=df), \
         mock.patch('pathlib.Path.exists', return_value=True):
        train, valid = pipeline.prepare_dataset('dummy.tsv')
        
    assert train['year'].max() < 2020
    assert valid['year'].min() >= 2020
    assert len(train) == 2
    assert len(valid) == 2

def test_tokenizer():
    tokenizer = GeneTokenizer(max_len=5)
    genes = pl.Series('genes', [['A', 'B'], ['B', 'C'], ['D']])
    tokenizer.fit(genes)
    
    assert 'A' in tokenizer.vocab
    assert 'B' in tokenizer.vocab
    assert tokenizer.get_vocab_size() == 6 # <PAD>, <UNK>, A, B, C, D
    
    encoded = tokenizer.encode(['A', 'Z', 'C'])
    assert encoded[0] == tokenizer.vocab['A']
    assert encoded[1] == 1 # <UNK>
    assert encoded[2] == tokenizer.vocab['C']
    assert encoded[3] == 0 # <PAD>
    assert len(encoded) == 5
