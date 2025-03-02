import os
import torch
import transformers

from tqdm import tqdm
from .ud_data import get_dataloaders
from .parser import MLP
from .pos_train import calculate_acc

def evaluate(data_dir, batch_size, load_name):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    _, _, test_loader = get_dataloaders(data_dir, batch_size, num_workers = 4)
    out_features = len(test_loader.dataset.pos2idx)
    
    bert_model = transformers.AutoModel.from_pretrained('bert-base-uncased', cache_dir = '/storage/vsub851/.cache')
    model = MLP(bert_model = bert_model, in_features = 768, hidden_state = 5, out_features = out_features)
    model = model.to(device)

    model.load_state_dict(torch.load(f'/storage/vsub851/multimeter-interp/language_modeling/syntax_parser/pos_models/{load_name}.pt'))

    total_acc = 0
    num_batches = 0
    for batch in tqdm(test_loader, desc = 'Iterating over test batches...'):
        input_ids = batch['input_ids'].to(device)
        head_indices = batch['pos_indices'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        logits, _, _ = model.forward(input_ids, attention_mask)

        mask = attention_mask.view(-1).bool()
        batch_size, seq_len, _ = logits.shape
        logits = logits.view(batch_size * seq_len, -1)
        labels = head_indices.view(-1)

        predictions = logits.argmax(dim=1)
        acc = calculate_acc(predictions, labels, attention_mask.view(-1).bool())
        total_acc += acc
        num_batches += 1
    test_avg_acc = total_acc / num_batches
    return test_avg_acc

if __name__ == '__main__':
    print(evaluate('/storage/vsub851/multimeter-interp/multi-datasets/ud-treebanks', 32, 'bert-pos-parser'))