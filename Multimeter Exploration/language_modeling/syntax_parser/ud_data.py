import os
import torch
import torch.utils.data as data
import conllu
import transformers

class UDDataset(data.Dataset):
    def __init__(self, data_dir, split = 'train', max_length = 512, pos2idx = {}):
        self.split = split
        self.data_file = os.path.join(data_dir, 'UD_English-EWT', f'en_ewt-ud-{split}.conllu')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir = '/storage/vsub851/.cache')
        self.max_length = max_length

        self.sentences = []
        self.pos2idx = pos2idx
        self.parse_conllu_file()

    def parse_conllu_file(self):
        if not self.pos2idx:
            self.pos2idx = {'UNK': 0}
        with open(self.data_file, 'r', encoding = 'utf-8') as f:
            for tokenlist in conllu.parse_incr(f):
                words = [token['form'] for token in tokenlist]
                heads = [token['head'] for token in tokenlist]
                pos = [token['upos'] for token in tokenlist]
                self.sentences.append((words, heads, pos))
                for p in pos:
                    if p not in self.pos2idx and self.split == 'train':
                        self.pos2idx[p] = len(self.pos2idx)
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        words, heads, pos_tags = self.sentences[idx]

        encoded = self.tokenizer(words, 
                                 is_split_into_words=True, 
                                 return_tensors='pt', 
                                 padding='max_length', 
                                 max_length=self.max_length, 
                                 truncation=True)

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        word_ids = encoded.word_ids()

        head_indices = torch.zeros(self.max_length, dtype = torch.long)
        last_word_idx = -1
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None:
                if word_idx != last_word_idx:
                    head = heads[word_idx]
                    if head == None:
                        head_indices[token_idx] = self.max_length - 1
                    elif head > 0:
                        head_token_idx = word_ids.index(head - 1)
                        head_indices[token_idx] = head_token_idx
                    else:
                        head_indices[token_idx] = 0
                else:
                    head_indices[token_idx] = head_indices[token_idx - 1]
                last_word_idx = word_idx
            else:
                head_indices[token_idx] = token_idx

        pos_indices = torch.zeros(self.max_length, dtype = torch.long)
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None:
                if word_idx != last_word_idx:
                    pos = pos_tags[word_idx]
                    try:
                        pos_indices[token_idx] = self.pos2idx[pos]
                    except:
                        pos_indices[token_idx] = self.pos2idx['UNK']
                else:
                    pos_indices[token_idx] = pos_indices[token_idx - 1]
                last_word_idx = word_idx
            else:
                pos_indices[token_idx] = self.pos2idx['UNK']
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'head_indices': head_indices, 'pos_indices': pos_indices}

def get_dataloader(data_dir, split, batch_size, num_workers, pos2idx = {}):
    dataset = UDDataset(data_dir, split, max_length = 128, pos2idx = pos2idx)
    return data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, shuffle = (split == 'train'))

def get_dataloaders(data_dir, batch_size, num_workers):
    loaders = []
    for split in ['train', 'dev', 'test']:
        if split == 'train':
            train_loader = get_dataloader(data_dir, split, batch_size = batch_size, num_workers = num_workers)
            loaders.append(train_loader)
        else:
            loader = get_dataloader(data_dir, split, batch_size = batch_size, num_workers = num_workers, pos2idx = train_loader.dataset.pos2idx)
            loaders.append(loader)
    return loaders

if __name__ == '__main__':
    ud_dataset = UDDataset('../../multi-datasets/ud-treebanks')
    ud_loader = data.DataLoader(ud_dataset, batch_size = 32)
    for batch in ud_loader:
        print(batch['input_ids'].shape, batch['head_indices'].shape, batch['pos_indices'].shape)
        break