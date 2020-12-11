# Titles

- Build Natural Language Processing products with PyTorch, FastAPI and Docker.

    - Build and deploy  Natural Language Processing deep-learning models
    
    - Practical Natural Language Processing: from development to deployment.
        - fastest path to building real-world solutions using NLP
  

## Target audience

- Beginner data scientists who have trained at least one neural network
model in the past using Python PyTorch or TensorFlow.

## Requirements
- Comfortable in Python.
- Basic knowledge of any Python deep-learning framework like PyTorch or Tensorflow.


## Introduction

- Are you interested in building and shipping NLP solutions using deep-learning and software
development best practices?

- Do you want to become an NLP engineer, able to train and deploy NLP models
using deep-learning?

- Do you want to become a professional Natural Language Processing engineer?

This course focuses on building NLP products, from training to deployment, using
PyTorch, FastAPI and Docker.

Very practical course. At the end of this course you will be able 
to showcase a portfolio of production ready deep-learning models.


Businesses do not look for ML researchers to build prototypes in Jupyter notebooks.
A key to professional success for ML engineers is the ability to 

Such profile is in high-demand in all industries.

It uniquely combines several key skills:

0. Machine learning fundamentals.
1. Machine learning model training, either from scratch or by fine-tuning a pre-trained model.
    This is the aspect most courses focus on. It is a critical one yes, and we will
    spend a fair amount of time discussing diferent modelling aproaches. However, it
    is not the last step in the ML cycle in the real-world.
    
2. Model API building
3. Model API containerization and deployment onto a cloud provider like Amazon Web Services or Google Cloud.

real-world applications of NLP models. To successfully apply
cutting-edge NLP models for your problems, you need to combine a few different skills:


To successfully apply NLP in real-world applications, you need to have the skills
to use all the great ML research publicly available efficiently.

    This course is addressed to data scientists and machine learning engineers who
    need to learn quickly how to take advantage of cutting edge NLP research
    on their workplace.



**Note**: This is not a course for beginners in Machine Learning.
I assume the following:
- You know what a neural network is.
- You are familiar with the Python language. You do not need to
be proficient.
- You have implemented at least one neural network model yourself (even a simple 1-layer
dense network) using either PyTorch of Tensorflow. You do not need to be proficient
in any of these 2 frameworks.


## Course outline: WIP

### 0. Introduction

- Big picture of the course

- Challenges that DS and ML engineers face in the real-world:
    - How can I move this prototype model from a Jupyter notebook into production?

### 1. [Optional] Review

- What has happened in NLP in the last 10 years?
    - Word embeddings: word2vec, GloVe
    - Sentence/document embeddings
    - LSTM
    - ELMO: Bidirectional LSTM
    - The Transformer
    - OpenAIGPT
    - BERT
    - OpenAIGPT2

- Deep learning frameworks and libraries.

- Review: how to train models with PyTorch
    - animated gif showing the train loop, top-down approach.


### 2. [Project] Build and deploy an email/SMS spam detector

- [ ] Plan of attack
    - Tools we are going to use: poetry, git, Jupyter, code editor (PyCharm, VSCode...)
    - Install necessary dependencies and build a model with Jupyter.
    - 
    
- [x] Train model 1: Bag-of-words with learned embeddings.
    - [Pending to add logging of embeddings with Tensorboard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
- [ ] Train model 2: Bag-of-words with GloVe embeddings.
- [ ] Train model 3: CNN with GloVe embeddings
- [ ] Train model 4: Dense network with BERT sentence embeddings as features
- [ ] Train model 5: Fine-tuning BERT
- [ ] Build model REST API with FastAPI  
- [ ] Containerize API with Docker
- [ ] Deploy container to Amazon EC2 instance.
- [ ] Deploy container to Amazon EC2 as a lambda function. 


## 6. Optimize models for inference/deployment
Goal: accuracy and speed.

Topics:
- Quantization


-----
# Material preparation

- [Slides](https://docs.google.com/presentation/d/1O3fpGM_AD3inJYeLqCaiTugfh0DIPtdnqJFIZYqpLis/edit#slide=id.gad8b967a81_0_0)

## Fundamentals of modern NLP
- [x] [Word embeddings from small labeled dataset in TF2](https://www.tensorflow.org/tutorials/text/word_embeddings)
- [x] [Word embeddings from small labeled dataset in PyTorch](http://localhost:8888/notebooks/notebooks/0_Word_embeddings_pytorch.ipynb)
- [ ] [Word2Vec in TF2](https://www.tensorflow.org/tutorials/text/word2vec)

## Real-world applications
- [ ] [Chat bots](https://arxiv.org/pdf/2004.13637.pdf)
    - https://parl.ai/docs/tasks.html
- [ ] [T5 fine tunning](https://github.com/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb)
- [ ] [Tacotron for text-to-speech](https://colab.research.google.com/gist/sayakmisra/2bf6e72fb9eed2f8cfb2fb47143726b6/-e2e-tts.ipynb#scrollTo=fjJ5zkyaoy29)

## Deployment of ML models
- [ ] [Deploy BERT sentiment analyzer with FastAPI](https://curiousily.com/posts/deploy-bert-for-sentiment-analysis-as-rest-api-using-pytorch-transformers-by-hugging-face-and-fastapi/)
- [ ] [fastAPI ML skeleton](https://github.com/cosmic-cortex/fastAPI-ML-quickstart/blob/master/api/ml/model.py)
- [ ] [FastAPI + AWS Lambda](https://towardsdatascience.com/fastapi-aws-robust-api-part-1-f67ae47390f9)
- [ ] [FastAPI + Docker + EC2](https://towardsdatascience.com/deployment-could-be-easy-a-data-scientists-guide-to-deploy-an-image-detection-fastapi-api-using-329cdd80400)

- [ ] [Custom dataset loading with torchtext](https://www.youtube.com/watch?v=0JOZt9xuRJM)
- [ ] [Custom dataset loading with torchtext](https://github.com/AI-Core/tutorials/blob/master/TorchText/torchtext_intro.ipynb)

- [ ] [TorchServe](https://github.com/pytorch/serve/blob/master/README.md#serve-a-model)



------
# Backlog

- [x] Add tensorboard in Colab version.
- [x] Enable notebooks as Colab notebooks in github

# References
- [Sentence embeddings with Siamese BERT](https://huggingface.co/sentence-transformers/bert-base-nli-cls-token)



# Calaix de sastre

- Initial title: Practical NLP course: from zero to hero.
It is too ambitious.

- https://docs.google.com/presentation/d/1O3fpGM_AD3inJYeLqCaiTugfh0DIPtdnqJFIZYqpLis/edit#slide=id.gad8b967a81_0_0