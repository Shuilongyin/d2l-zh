import torch

def prepare_input(cfg, text):
    '''
    tokenizer("Using a Transformer network is simple")
    {'input_ids': [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102],
     'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
     'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
    '''
    inputs = cfg.tokenizer(text,
                           add_special_tokens=True,
                           max_length=cfg.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long) # 转成tensor
    return inputs


