import sys,os
# sys.path.append('../')
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))

from utils.data import get_dataset
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np

def eval_ppl_wikitext(model, tokenizer, device, dataset=None, debug=False, context_size=2048, bs=1):
    """
    Evaluate perplexity (ppl) specifically on the wikitext dataset.

    Args:
        model (torch.nn.Module): The language model to be evaluated.
        testenc (TokenizerWrapper): Encoded input IDs from test set.
        bs (int): Batch size for evaluation.
        device (torch.device): Device to move data onto (e.g., 'cuda:0' or 'cpu').

    Returns:
        float: The perplexity of the language model on the wikitext test dataset.
    """
    text = ""
    for sample in dataset:
        text += sample["text"] + "\n\n"

    testenc = tokenizer(text, return_tensors="pt")

    # Get input IDs from the TokenizerWrapper instance
    testenc = testenc.input_ids

    model.seqlen = context_size

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    infer_sparsity_list = []
    # Loop through each batch
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)
        
        # Forward pass through the model
        lm_logits = model(inputs).logits    

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()    # Example: [cat, sat, on, ???] -> [cat, sat, on]
        shift_labels = inputs[:, 1:]    # Example: [The, cat, sat, on] -> [cat, sat, on]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.float()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        # print('loss:', loss)

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)    # nll = loss * sequence_length * batch_size


        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))    # ppl = exp(∑(nlls) / (num_samples * sequence_length))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

def eval_ppl_wikitext_with_inference_sparsity(model, tokenizer, device, dataset=None, debug=False, context_size=2048, bs=1):
    """
    Evaluate perplexity (ppl) specifically on the wikitext dataset.

    Args:
        model (torch.nn.Module): The language model to be evaluated.
        testenc (TokenizerWrapper): Encoded input IDs from test set.
        bs (int): Batch size for evaluation.
        device (torch.device): Device to move data onto (e.g., 'cuda:0' or 'cpu').

    Returns:
        float: The perplexity of the language model on the wikitext test dataset.
    """
    text = ""
    for sample in dataset:
        text += sample["text"] + "\n\n"

    testenc = tokenizer(text, return_tensors="pt")

    # Get input IDs from the TokenizerWrapper instance
    testenc = testenc.input_ids

    model.seqlen = context_size

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    infer_sparsity_list_attn_h1 = []
    infer_sparsity_list_attn_h2 = []
    infer_sparsity_list_mlp_h1 = []
    infer_sparsity_list_mlp_h2 = []
    # Loop through each batch
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)
        
        # Forward pass through the model
        lm_logits = model(inputs).logits    
        for layer in model.model.layers:
            infer_sparsity_list_attn_h1.append(layer.mlp.infer_sparsity_h1)
            infer_sparsity_list_attn_h2.append(layer.mlp.infer_sparsity_h2)
            infer_sparsity_list_mlp_h1.append(layer.self_attn.infer_sparsity_h1)
            infer_sparsity_list_mlp_h2.append(layer.self_attn.infer_sparsity_h2)

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()    # Example: [cat, sat, on, ???] -> [cat, sat, on]
        shift_labels = inputs[:, 1:]    # Example: [The, cat, sat, on] -> [cat, sat, on]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.float()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        # print('loss:', loss)

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)    # nll = loss * sequence_length * batch_size

        infer_sparsity_array = np.array(infer_sparsity_list_mlp_h1)
        print('model level mlp h1 inference sparsity:', np.mean(infer_sparsity_array))
        infer_sparsity_array = np.array(infer_sparsity_list_mlp_h2)
        print('model level mlp h2 inference sparsity:', np.mean(infer_sparsity_array))

        infer_sparsity_array = np.array(infer_sparsity_list_attn_h1)
        print('model level attn h1 inference sparsity:', np.mean(infer_sparsity_array))
        infer_sparsity_array = np.array(infer_sparsity_list_attn_h2)
        print('model level attn h2 inference sparsity:', np.mean(infer_sparsity_array))

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)
    
    infer_sparsity_array = np.array(infer_sparsity_list_mlp_h1)
    print('mlp h1 {:.2f}'.format(np.mean(infer_sparsity_array) * 100))
    infer_sparsity_array = np.array(infer_sparsity_list_mlp_h2)
    print('mlp h2 {:.2f}'.format(np.mean(infer_sparsity_array) * 100))

    infer_sparsity_array = np.array(infer_sparsity_list_attn_h1)
    print('attn h1 {:.2f}'.format(np.mean(infer_sparsity_array) * 100))
    infer_sparsity_array = np.array(infer_sparsity_list_attn_h2)
    print('attn h2 {:.2f}'.format(np.mean(infer_sparsity_array) * 100))

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))    # ppl = exp(∑(nlls) / (num_samples * sequence_length))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

def eval_ppl(model, tokenizer, device, dataset=None, debug=False, context_size=2048, window_size=512):
    text = ""
    for sample in dataset:
        text += sample["text"] + "\n\n"

    encodings = tokenizer(text, return_tensors="pt")

    if debug:
        print(tokenizer.decode(encodings.input_ids[0][:100]))

    max_length = context_size + window_size
    stride = window_size
    seq_len = encodings.input_ids.size(1)
    # make seq_len a multiple of stride
    seq_len = seq_len - (seq_len % stride)

    if debug:
        print(f"seq_len: {seq_len}")

    if debug:
        pbar = tqdm(range(0, seq_len, stride))
    else:
        pbar = range(0, seq_len, stride)

    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)

    if model_vocab_size != tokenizer_vocab_size:
        # resize model embeddings to fit tokenizer
        if model_vocab_size != tokenizer_vocab_size:
            print("Resize model embeddings to fit tokenizer")
        model.resize_token_embeddings(tokenizer_vocab_size)

    model.eval()
    nlls = []

    for begin_loc in pbar:
        end_loc = begin_loc + max_length
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-stride] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # print(outputs.logits.shape)
            neg_log_likelihood = outputs.loss

        if debug:
            pbar.set_description(
                f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
            )

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc >= seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).double().mean())

    return ppl.item()