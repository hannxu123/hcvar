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
## To run the code
An example code to run the code to train a RoBERTa-base classification model and test the model on the domain "review".
```ruby
python -m detector.train_binary_cls --domain review 
```

## To evaluate the generalization of detectors
1. Load dataset

2. XXX the model

3. Train the model
