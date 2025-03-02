import os
import numpy as np
import torch
import transformers

def get_leaf_modules(module, leaf_modules = []):
    if not list(module.children()):
        leaf_modules.append(module)
    for child in module.children():
        get_leaf_modules(child, leaf_modules)
    
    return leaf_modules

def get_module_weights(module):
    weights = []
    for name, param in module.named_parameters():
        if 'weight' in name:
            weights.append(param.data)
    return weights

def calculate_average_magnitude(weights):
    total_magnitude = 0
    total_elements = 0
    for weight in weights:
        total_magnitude += torch.sum(torch.abs(weight))
        total_elements += weight.numel()
    return (total_magnitude/total_elements).item()

def get_param(model_path):
    assert os.path.exists(os.path.join(model_path, 'parameter.pt'))
    param = torch.load(os.path.join(model_path, 'parameter.pt'))
    return torch.sigmoid(param)

if __name__ == '__main__':
    model_path = os.path.join('language_modeling', 'new-gpt2-wikitext-multi-deps', 'checkpoint-10000')
    model = transformers.AutoModel.from_pretrained(os.path.join(model_path, 'original_network'))

    leaf_modules = get_leaf_modules(model)
    start_index = 65
    end_index = 68

    avg_magnitudes = []
    for i in range(start_index, end_index):
        module = leaf_modules[i]
        weights = get_module_weights(module)
        if not weights:
            continue
        avg_magnitude = calculate_average_magnitude(weights)
        avg_magnitudes.append(avg_magnitude)
    print(np.mean(avg_magnitudes))

    get_param(model_path)
