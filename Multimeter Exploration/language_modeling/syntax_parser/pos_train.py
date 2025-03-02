import torch
import torch.nn as nn
import torch.optim as optim
import transformers

from tqdm import tqdm
from .ud_data import get_dataloaders
from .parser import MLP

def calculate_acc(predictions, gold_heads, mask):
    correct = ((predictions == gold_heads) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0

def train(data_dir, batch_size, lr, num_epochs, save_name):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_loader, dev_loader, _ = get_dataloaders(data_dir, batch_size, num_workers = 4)

    num_pos = len(train_loader.dataset.pos2idx)

    bert_model = transformers.AutoModel.from_pretrained('bert-base-uncased', cache_dir = '/storage/vsub851/.cache')
    model = MLP(bert_model = bert_model, in_features = 768, out_features = num_pos, hidden_state = 5)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    for epoch in range(num_epochs):
        model = model.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        for batch in tqdm(train_loader, desc = f'Training Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            pos_indices = batch['pos_indices'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            logits, _, _ = model.forward(input_ids, attention_mask)

            mask = attention_mask.view(-1).bool()
            batch_size, seq_len, _ = logits.shape
            logits = logits.view(batch_size * seq_len, -1)
            labels = pos_indices.view(-1)
            loss = loss_fn(logits[mask], labels[mask])

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            acc = calculate_acc(predictions, labels, attention_mask.view(-1).bool())
            total_acc += acc
            num_batches += 1

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / num_batches
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}')

        model = model.eval()
        dev_total_loss = 0
        dev_total_acc = 0
        dev_num_batches = 0

        for batch in tqdm(dev_loader, desc = f'Validation Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            pos_indices = batch['pos_indices'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits, _, _ = model.forward(input_ids, attention_mask)

            mask = attention_mask.view(-1).bool()
            batch_size, seq_len, _ = logits.shape
            logits = logits.view(batch_size * seq_len, -1)
            labels = pos_indices.view(-1)
            loss = loss_fn(logits[mask], labels[mask])

            dev_total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            uas = calculate_acc(predictions, labels, attention_mask.view(-1).bool())
            dev_total_acc += uas
            dev_num_batches += 1

        dev_avg_loss = dev_total_loss / len(dev_loader)
        dev_avg_acc = dev_total_acc / dev_num_batches
        scheduler.step(dev_avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {dev_avg_loss:.4f}, Val Accuracy: {dev_avg_acc:.4f}')

    torch.save(model.state_dict(), f'pos_models/{save_name}.pt')
    return model

if __name__ == '__main__':
    train('../multi-datasets/ud-treebanks', 32, 1e-4, 50, 'bert-pos-parser') 