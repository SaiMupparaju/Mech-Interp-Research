import os
import torch
import torch.nn as nn
import transformers
import datasets

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
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss += self.sparsity_lambda * l1_norm

        return (loss, outputs) if return_outputs else loss

def train(output_dir, use_sparse = True, num_train_epochs = 5, lr = 3e-5, per_device_train_batch_size = 6, save_steps = 10000, logging_dir = './logs', logging_steps = 500):
    # device = torch.device('cuda')
    dataset = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir = '/storage/vsub851/.cache')

    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir = '/storage/vsub851/.cache')
    tokenizer.pad_token = tokenizer.eos_token
    # Function to add labels for language modeling
    def prepare_data(examples):
        tokenized_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
        tokenized_inputs['labels'] = tokenized_inputs['input_ids'].copy()
        return tokenized_inputs

    dataset = dataset.map(prepare_data, batched=True)
    
    gpt2_config = transformers.GPT2Config.from_pretrained('gpt2', cache_dir = '/storage/vsub851/.cache')
    model = transformers.AutoModelForCausalLM.from_config(gpt2_config)

    training_args = transformers.TrainingArguments(
        output_dir = output_dir,
        overwrite_output_dir = True,
        num_train_epochs = num_train_epochs,
        per_device_train_batch_size = per_device_train_batch_size,
        learning_rate = lr,
        save_steps = save_steps,
        logging_dir = logging_dir,
        logging_steps = logging_steps,
        do_eval = True,
        eval_strategy = 'steps',
        eval_steps = 5000
    )

    if not use_sparse:
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            tokenizer=tokenizer,
        )
    # use sparse trainer instead of normal trainer. We want to ablate the weights early.
    else:
        trainer = SparseTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            tokenizer=tokenizer,
        )
    trainer.train()

if __name__ == '__main__':
    train('./sparse-gpt2-wikitext')