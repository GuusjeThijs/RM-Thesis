import torch
from transformers import BertForMaskedLM, BertTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import numpy as np
import sys


def main(filename):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Load the pre-trained BERT model and tokenizer
    print('loading pretrained model...')
    model = BertForMaskedLM.from_pretrained('GroNLP/bert-base-dutch-cased')
    tokenizer = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, mlm=True, return_tensors='pt')
    
    with open(filename, 'r') as rf:
        lines = rf.readlines()
        
    length_corpus = len(lines)
    print(length_corpus)

    print('loading done')

    # # Load your custom dataset
    dataset = load_dataset('text', data_files={'train':[filename]}, split='train', streaming=True)
    tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=32, padding='max_length', return_tensors='pt'), batched=True, batch_size=128)
    print(dataset)
    print(tokenized_dataset)

    # Define hyperparameters
    batch_size = 128
    learning_rate = 2e-5
    num_epochs = 2

    training_args = TrainingArguments(
        output_dir="output",
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        max_steps= np.ceil(length_corpus/batch_size)*num_epochs,
        logging_steps=10000,
        save_steps=np.ceil(length_corpus/batch_size)/2
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    trainer.train()

    trainer.save_model("Finetuned")

if __name__ == '__main__':
    filename = sys.argv[1]
    main(filename)