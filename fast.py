import torch
import config
from fastapi import FastAPI
from model import SentimentRNN
from utils import predict,tokenizer
from flask import Flask, jsonify, request
import requests
from vocabtoint import vocab_to_int
import torch
import torch.nn as nn

app = FastAPI()


vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
net.load_state_dict(torch.load('tmp.pt',map_location=torch.device('cpu')))

# net = torch.load('entire_model.pt')

net.eval()

@app.get("/ping")
def ping():
    return {"message": "hello!"}


@app.get("/predict/{sentence}")
def prediction(sentence: str):
    pred = predict(net,sentence)
    return pred