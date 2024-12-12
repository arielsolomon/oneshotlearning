from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = 'cuda'
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
big_model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", torch_dtype=torch.bfloat16).to(device)
small_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", torch_dtype=torch.bfloat16).to(device)
