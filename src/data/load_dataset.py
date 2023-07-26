# -*- coding: utf-8 -*-
import datasets
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os


def _download_tokenizer(tokenizer_path_or_name, seq_length, cache_dir=None):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_name, cache_dir=cache_dir)
        tokenizer.model_max_length = seq_length
    except OSError as error_msg:
        raise OSError(f"Invalid huggingface tokenizer {tokenizer_path_or_name} given: {error_msg}")
    return tokenizer



def load_tokenizer(tokenizer_path_or_name, seq_length=512, vocab_size=None, cache_dir=None):
    """Load a tokenizer from disk/huggingface. This will never construct a new tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_name, model_max_length=seq_length)
    except FileNotFoundError:
        tokenizer = _download_tokenizer(tokenizer_path_or_name, seq_length, cache_dir)
    if vocab_size is not None and tokenizer.vocab_size != vocab_size:
        raise ValueError(f"Loaded tokenizer with vocab_size {tokenizer.vocab_size} incompatible with given vocab.")
    return tokenizer



def load_pretraining_corpus(cfg, data_path):

    print("load pretraining corpus from huggingface")

    tokenized_dataset = datasets.load_dataset(cfg.hf_location, "train", streaming=cfg.streaming, cache_dir=data_path)["train"]
    tokenized_dataset = tokenized_dataset.with_format("torch")

    tokenizer_req_files = ["special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]
    os.makedirs(os.path.join(data_path, "tokenizer"), exist_ok=True)
    for file in tokenizer_req_files:
        hf_hub_download(
            cfg.hf_location,
            file,
            subfolder="tokenizer",
            repo_type="dataset",
            local_dir=os.path.join(data_path),
        )
    tokenizer = load_tokenizer(os.path.join(data_path, "tokenizer"), seq_length=cfg.seq_length, cache_dir=data_path)
    return tokenized_dataset, tokenizer



if __name__ == "__main__":

    ## in config ##
    ## dict does not work ##
    data_path = '~/.cache/huggingface/datasets'
    dummy_cfg = {'hf_location' : 'JonasGeiping/the_pile_WordPiecex32768_2efdb9d060d1ae95faf952ec1a50f020',
                 'streaming': False,
                 'seq_length': 128}
    
    dataset, tokenizer = load_pretraining_corpus(dummy_cfg, data_path)
    print(dataset)


