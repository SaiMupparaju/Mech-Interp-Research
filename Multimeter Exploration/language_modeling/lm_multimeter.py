import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import datasets

from syntax_parser import PairwiseMLP, MLP
sys.path.append('../')
from hf_multimeter import HFCombinedNetwork

class SparseTrainer(transformers.Trainer):
    def __init__(self, *args, sparsity_lambda=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparsity_lambda = sparsity_lambda

    def compute_loss(self, model, inputs, return_outputs=False):
        # Compute the original loss
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        lm_logits = outputs.logits
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Add L1 regularization for sparsity
        l1_norm = sum(p.abs().sum() for p in model.original_network.parameters())
        loss += self.sparsity_lambda * l1_norm

        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir = None, state_dict = None):
        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok = True)
        model = self.model_wrapped if hasattr(self, 'model_wrapped') else self.model
        model.original_network.save_pretrained(os.path.join(output_dir, 'original_network'))

        torch.save(model.frozen_network, os.path.join(output_dir, 'frozen_network'))
        torch.save(model.parameter, os.path.join(output_dir, 'parameter.pt'))

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        if self.optimizer and self.lr_scheduler:
            torch.save(
                {
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                },
                os.path.join(output_dir, 'optimizer.pt'),
            )

        self.save_state()

class SyntaxFrozenNetwork(nn.Module):
    def __init__(self, in_features, model_in_features, model_out_features, out_features, model):
        super(SyntaxFrozenNetwork, self).__init__()
        self.linear1 = nn.Linear(in_features, model_in_features)
        self.linear2 = nn.Linear(model_out_features, out_features)
        self.relu = nn.ReLU()

        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        # print(x.shape)
        x = self.linear1(x)
        x, _, _ = self.model(inputs = None, attention_mask = None, inputs_embeds = x)
        #NOTE: A quick note to self; we choose to send logits instead of representations for the reason that 
        # representations may have added information/capacity when using BERT. It's super important to NOT
        # let this added capacity ablate more weights of the network. So we only use the logits instead.
        x = x.view(x.shape[0], x.shape[1] * x.shape[2])
        x = self.relu(self.linear2(x))
        x = x.view(x.shape[0], -1, 768)
        return x
    
def lm_multimeter(output_dir, pretrained_dir, mul_network, frozen_dir, index1, index2, use_pretrained = True, num_train_epochs = 10, lr = 3e-5, per_device_train_batch_size = 4, save_steps = 5000, logging_dir = './logs', logging_steps = 500):
    # Set up original network
    if use_pretrained:
        gpt2_model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_dir)
    else:
        # Sanity Check: train from a randomly initialized network instead
        gpt2_config = transformers.AutoConfig.from_pretrained('gpt2', cache_dir = '/storage/vsub851/.cache')
        gpt2_model = transformers.AutoModelForCausalLM.from_config(gpt2_config)

    # Set up frozen network
    bert_model = transformers.AutoModel.from_pretrained('bert-base-uncased', cache_dir = '/storage/vsub851/.cache')
    # NOTE: 128 refers to the max length from the model. TODO: Add these as parameters
    if mul_network == 'dep':
        dep_model = PairwiseMLP(bert_model, in_features = 768, hidden_state = 5)
        dep_model.load_state_dict(torch.load(frozen_dir))
        frozen_network = SyntaxFrozenNetwork(in_features = 768, model_in_features = 768, model_out_features = 128 * 128, out_features = 128 * 768, model = dep_model)
    elif mul_network == 'pos':
        ckpt = torch.load(frozen_dir)
        num_pos = ckpt['linear2.weight'].shape[0]
        pos_model = MLP(bert_model, in_features = 768, hidden_state = 5, out_features = num_pos)
        pos_model.load_state_dict(torch.load(frozen_dir))
        frozen_network = SyntaxFrozenNetwork(in_features = 768, model_in_features = 768, model_out_features = 128 * num_pos, out_features = 128 * 768, model = pos_model)
    else:
        raise NotImplementedError

    # Set up combined network
    combined_network = HFCombinedNetwork(gpt2_model, frozen_network, representation_layer_index = index1, modification_layer_index = index2)

    # Set up dataset from wikitext again
    dataset = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir = '/storage/vsub851/.cache')
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(pretrained_dir)
    tokenizer.pad_token = tokenizer.eos_token
    # Function to add labels for language modeling
    def prepare_data(examples):
        tokenized_inputs = tokenizer(examples['text'], truncation = True, padding = 'max_length', max_length = 128)
        tokenized_inputs['labels'] = tokenized_inputs['input_ids'].copy()
        return tokenized_inputs

    dataset = dataset.map(prepare_data, batched=True)

    training_args = transformers.TrainingArguments(
        output_dir = output_dir,
        overwrite_output_dir = True,
        num_train_epochs = num_train_epochs,
        per_device_train_batch_size = per_device_train_batch_size,
        learning_rate = lr,
        save_steps = save_steps,
        logging_dir = logging_dir,
        logging_steps = logging_steps
    )

    trainer = SparseTrainer(
        model=combined_network,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
    )

    print('Beginning training...')
    trainer.train()

if __name__ == '__main__':
    lm_multimeter(
        output_dir = 'sparse_init-gpt2-wikitext-multi-pos',
        pretrained_dir = 'sparse-gpt2-wikitext/checkpoint-30000',
        mul_network = 'dep',
        frozen_dir = 'syntax_parser/dep_models/bert-dep-parser.pt',
        index1 = 45, 
        index2 = 48,
        use_pretrained = True
    )