# HC-Var (Human and ChatGPT texts with Variety)
This is a repository for training binary classifcation models to distinguish human texts and ChatGPT (GPT3.5-Turbo) generated texts.
We collect a new dataset HC-Var (Human and ChatGPT texts with Variety) to fulfill our objective. 
This dataset includes the texts which are generated / human written to accomplish various language tasks with various approaches. 
The included language tasks and topics are summarized below. 
The HC-Var dataset is available now in hugging face: https://huggingface.co/datasets/hannxu/hc_var. 

## Dataset Summary
This dataset contains human and ChatGPT texts to fulfill 4 distinct language tasks, including news composing (News), review (Review), essay writing (Writing) and question answering (QA). Under each task, 
we collect the human and ChatGPT generated texts with one or multiple topics. 
For each language task, this dataset considers 3 different prompts to inquire ChatGPT outputs. 

| Domain (Task) | News   | News   | News     | Review | Review | Writing  | QA      | QA      | QA      | QA      |
|---------------|--------|--------|----------|--------|--------|----------|---------|---------|---------|---------|
| Topic         | World  | Sports | Business | IMDb   | Yelp   | Essay    | Finance | Histroy | Medical | Science |
| ChatGPT Vol.  | 4,500  | 4,500  | 4,500    | 4,500  | 4,500  | 4,500    | 4,500   | 4,500   | 4,500   | 4,500   |
| Human Vol.    | 10,000 | 10,000 | 9,096    | 10,000 | 10,000 | 10,000   | 10,000  | 10,000  | 10,000  | 10,000  |
| Human Source  | XSum   | XSum   | XSum     | IMDb   | Yelp   | IvyPanda | FiQA    | Reddit  | MedQuad | Reddit  |


## Enivornments
The code is primary runned and examined under python 3.10.12, torch 2.0.1. To install other required packages using the command:
```ruby
pip install -r requirements.txt
```
## To train the model
This repository currently supports training classification models under RoBERTa-base, RoBERTa-large and T5 (with various architectures, usually T5-small and T5-base).
An example command to run the code to train a RoBERTa-base classification model and test the model on the domain "review".
```ruby
python -m detector.train_binary_cls --domain review 
```
For details, the training process includes 3 major steps:
1. Load the training, validation and test dataloaders.
```ruby
train_loader, valid_loader, test_loader = Loader(batch_size = 32, domain=domain, cache_dir = cache_dir)
```
2. Initilize the classification model and optimizer:
```ruby
model_name = 'roberta-large' 
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)
```
3. Train the model.
```ruby
def train(model, tokenizer, optimizer, device, loader):
    model.train()
    for i, dat in enumerate(loader):
        texts, labels = dat
        texts = list(texts)
        result = tokenizer(texts, return_tensors="pt", padding = 'max_length', max_length = 256, truncation=True)
        texts, masks, labels = result['input_ids'].to(device), result['attention_mask'].to(device), labels.to(device)
        aa = model(texts, labels=labels, attention_mask = masks)
        loss = aa['loss']
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```
## To evaluate the model
We define 3 types of test data loaders to evaluate the models performance facing different varieties.
For example, to evaluate a model's performance when test samples are divided in different tasks:
```ruby
test_loader = Domain_loader(domain= "TaskName", cache_dir = cache_dir)  ## TaskName can be News, Review, Writing, QA
```
Or when test samples are divided in different topics in the same task, i.e., QA:
```ruby
test_loader = Topic_loader(domain= 'QA', topic = "TopicName", cache_dir = cache_dir)  ## TopicName can be history, finance, medical, science
```
Or when test samples are divided in different prompts in the same task, i.e., QA:
```ruby
test_loader = Prompt_loader(domain= 'QA', prompt = promptid, cache_dir = cache_dir)  ## promptid can be "P1", "P2", "P3"
```

