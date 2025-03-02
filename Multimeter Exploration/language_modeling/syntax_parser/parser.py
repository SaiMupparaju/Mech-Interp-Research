import torch
import torch.nn as nn
import transformers

class PairwiseMLP(nn.Module):
    def __init__(self, bert_model, in_features, hidden_features = None, hidden_state = -1):
        super(PairwiseMLP, self).__init__()
        self.bert_model = bert_model
        self.hidden_state = hidden_state

        self.in_features = in_features
        self.hidden_features = hidden_features or in_features

        self.linear1 = nn.Linear(self.in_features * 2, self.hidden_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_features, 1)

    def forward(self, inputs, attention_mask, inputs_embeds = None):
        if inputs_embeds == None:
            inputs = self.bert_model(inputs, attention_mask, output_hidden_states = True)
        else:
            inputs = self.bert_model(inputs_embeds = inputs_embeds, output_hidden_states = True)
        inputs = inputs.hidden_states[self.hidden_state].squeeze()
        batch_size, length, features = inputs.shape
        lefts = inputs.unsqueeze(2).repeat(1, 1, length, 1)
        rights = inputs.unsqueeze(1).repeat(1, length, 1, 1)
        pairs = torch.cat((lefts, rights), dim=-1)
        pairs = pairs.view(batch_size, length*length, 2*features)
        out1 = self.relu(self.linear1(pairs))
        return self.linear2(out1).view(batch_size, length, length), out1, inputs
    
class MLP(nn.Module):
    def __init__(self, bert_model, in_features, out_features, hidden_features = None, hidden_state = -1):
        super(MLP, self).__init__()
        self.bert_model = bert_model
        self.hidden_state = hidden_state

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features or in_features

        self.linear1 = nn.Linear(self.in_features, self.hidden_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_features, self.out_features)
    
    def forward(self, inputs, attention_mask, inputs_embeds = None):
        if inputs_embeds == None:
            inputs = self.bert_model(inputs, attention_mask, output_hidden_states = True)
        else:
            inputs = self.bert_model(inputs_embeds = inputs_embeds, output_hidden_states = True)

        inputs = inputs.hidden_states[self.hidden_state].squeeze()
        out1 = self.relu(self.linear1(inputs))
        return self.linear2(out1), out1, inputs
        
if __name__ == '__main__':
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir = '/storage/vsub851/.cache')
    bert_model = transformers.AutoModel.from_pretrained('bert-base-uncased', cache_dir = '/storage/vsub851/.cache')

    model = PairwiseMLP(bert_model = bert_model, in_features = 768, hidden_state = 5)
    model = model.to('cuda')

    sent = 'This is a longer sentence I am using for debugging this stupid parser.'
    input_ids = tokenizer(sent, return_tensors = 'pt')['input_ids'].to('cuda')
    preds, out1 = model.forward(input_ids)
    print(preds.shape)