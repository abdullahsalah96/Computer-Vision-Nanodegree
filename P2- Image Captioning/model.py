import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def preset_hidden_state(self, seq_length):
        print(seq_length)
        print("hidden shape: " + str(self.num_layers) + "," + str(seq_length) + "," +  str(self.hidden_size))
        hidden = (
            torch.zeros(self.num_layers, seq_length, self.hidden_size),
            torch.zeros(self.num_layers, seq_length, self.hidden_size)
        )
        return hidden
        
    def forward(self, features, captions):
#         _ = preset_hidden_state(features.shape[0])
        hidden = self.preset_hidden_state(features.shape[0]) #initializing hidden state
        print(features.shape)
        print("hidden shape: " + str(hidden[0].shape))
        if(captions is not None):
            captions = captions[:,:-1] #ignoring end word
#         print("Features shape: " + str(features.shape)) 
        features = features.unsqueeze(1) #make input features of the same shape of the embedded layer output to concat them
#         print("Features shape After squeezing: " + str(features.shape))
#         print("Captions shape before embedding: " + str(captions.shape))
        embeds = self.embedding(captions) #embedding captions
#         print("Captions After embedding: " + str(captions.shape))
        embeds = torch.cat((features, embeds), dim = 1)
#         print(embeds.shape)
        out, hidden = self.lstm(embeds)
        print(out.shape())
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)"
        outputs = []
        hidden = self.preset_hidden_state(1) #initializing hidden state
#         print("Input shape: " + str(inputs.shape))
        with torch.no_grad():
            for i in range(max_len):
                out, states = self.lstm(inputs, states)
                out = self.fc(out).squeeze(0)
                val, index = out.max(1)
#                 print(out.shape)
#                 print("VAL shape: " + str(val.shape))
#                 print("VAL: " + str(val))
#                 print("INDEX shape: " + str(index.shape))
#                 print("INDEX: " + str(index.item()))
                outputs.append(index.item())
#                 print("To embedding shape: " + str(index.shape))
#                 print("To embedding: " + str(index))
#                 print("old inputs shape: " + str(inputs.shape))
#                 print("old inputs: " + str(inputs))
                inputs = self.embedding(index).unsqueeze(1)
#                 print("new inputs shape: " + str(inputs.shape))
#                 print("new inputs: " + str(inputs))
        return outputs