from tqdm import tqdm 
import torch 

def build_tokenizer_for_dataset(dataset): 
    tokens = set()
    for example in tqdm(dataset['train']):
        tokens.update(example['text'])
    tokens = list(tokens)
    tokenizer = {tok:i for (i,tok) in enumerate(tokens)}
    return tokenizer 

def tokenize(tokenizer, el): 
    return list(map(lambda x: tokenizer[x], el))

