root="./data_enhanced/"
with open(root+"log",'w+') as f : pass

from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch.utils.data import DataLoader as TorchDataLoader
from Model import Model
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
import time
import os
import pandas as pd
import random

t0=time.time()

CE = torch.nn.CrossEntropyLoss()
def contrastive_loss(v1, v2):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  labels = torch.arange(logits.shape[0], device=v1.device)
  return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

model_name = 'distilbert-base-uncased'

download_models = False
if download_models :
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(root+"tokenizer")
else :
    tokenizer = AutoTokenizer.from_pretrained(root+"tokenizer")

gt = np.load(root+"token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root=root, gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root=root, gt=gt, split='train', tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nb_epochs = 5
batch_size = 32
learning_rate = 2e-5

#construct the loaders
datas=[]
for cid in val_dataset.cids :
    data=val_dataset.get_cid(cid)
    data=Data(x=data.x, edge_index=data.edge_index,
              input_ids=data.input_ids, attention_mask=data.attention_mask)
    datas.append(data)
val_loader = DataLoader(datas, batch_size=batch_size, shuffle=True)

random.shuffle(train_dataset.cids)
text_dataset=[]
graph0_dataset=[]
graph1_dataset=[]
graph2_dataset=[]
for cid in train_dataset.cids :
    data=train_dataset.get_cid(cid)
    graph0_dataset.append(Data(x=data.x, edge_index=data.edge_index))
    graph1_dataset.append(Data(x=data.x1, edge_index=data.edge_index1))
    graph2_dataset.append(Data(x=data.x2, edge_index=data.edge_index2))
    text_dataset.append(Data(input_ids=data.input_ids, attention_mask=data.attention_mask))
   

if download_models :
    model = Model(model_name=model_name, num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
    model.text_encoder.bert.save_pretrained(root+"text_encoder")
else :
    model = Model(model_name=root+"text_encoder", num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
    
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                betas=(0.9, 0.999),
                                weight_decay=0.01)

epoch = 0
loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = 50
best_validation_loss = 1000000

for i in range(nb_epochs):
    print('-----EPOCH{}-----'.format(i+1))
    with open(root+"log",'a') as f :
        f.write(f"EPOCH{i+1}:{(time.time()-t0)//60}min\n")
    model.train()
    #reorder dataset
    order=list(range(N_train))
    random.shuffle(order)
    train_text_loader = DataLoader([text_dataset[order[x]] for x in range(N_train)], batch_size=batch_size, shuffle=False)
    train_graph0_loader = DataLoader([graph0_dataset[order[x]] for x in range(N_train)], batch_size=batch_size, shuffle=False)
    train_graph1_loader = DataLoader([graph1_dataset[order[x]] for x in range(N_train)], batch_size=batch_size, shuffle=False)
    train_graph2_loader = DataLoader([graph2_dataset[order[x]] for x in range(N_train)], batch_size=batch_size, shuffle=False)
  
    for text_batch, graph0_batch, graph1_batch, graph2_batch in\
    zip(train_text_loader,train_graph0_loader,train_graph1_loader,train_graph2_loader):
        #separate text features
        input_ids = text_batch.input_ids
        attention_mask = text_batch.attention_mask
        
        #compute outputs
        x_text = model.text_encoder(input_ids.to(device), attention_mask.to(device))
        x_graph0 = model.graph_encoder(graph0_batch.to(device))
        x_graph1 = model.graph_encoder(graph1_batch.to(device))
        x_graph2 = model.graph_encoder(graph2_batch.to(device))
        
        #compute loss
        current_loss = contrastive_loss(x_graph0, x_text)+\
                    contrastive_loss(x_graph1,x_graph2)

        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
        loss += current_loss.item()
        
        count_iter += 1
        if count_iter % printEvery == 0:
            time2 = time.time()
            print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                        time2 - time1, loss/printEvery))
            losses.append(loss)
            loss = 0 
    model.eval()       
    val_loss = 0        
    for batch in val_loader:
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        x_graph, x_text = model(graph_batch.to(device), 
                                input_ids.to(device), 
                                attention_mask.to(device))
        current_loss = contrastive_loss(x_graph, x_text)   
        val_loss += current_loss.item()
    best_validation_loss = min(best_validation_loss, val_loss)
    print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)) )
    if best_validation_loss==val_loss:
        print('validation loss improoved saving checkpoint...')
        save_path = os.path.join('./', 'model'+str(i)+'.pt')
        torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_accuracy': val_loss,
        'loss': loss,
        }, save_path)
        print('checkpoint saved to: {}'.format(save_path))

with open(root+"log",'a') as f :
    f.write(f"LEARNING DONE:{(time.time()-t0)//60}min\n")
print('loading best model...')
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

graph_model = model.get_graph_encoder()
text_model = model.get_text_encoder()

test_cids_dataset = GraphDataset(root=root, gt=gt, split='test_cids')
test_text_dataset = TextDataset(file_path=root+'test_text.txt', tokenizer=tokenizer)

idx_to_cid = test_cids_dataset.get_idx_to_cid()

test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)

graph_embeddings = []
for batch in test_loader:
    for output in graph_model(batch.to(device)):
        graph_embeddings.append(output.tolist())

test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)
text_embeddings = []
for batch in test_text_loader:
    for output in text_model(batch['input_ids'].to(device), 
                             attention_mask=batch['attention_mask'].to(device)):
        text_embeddings.append(output.tolist())


from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(text_embeddings, graph_embeddings)

solution = pd.DataFrame(similarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
solution.to_csv('submission.csv', index=False)

with open(root+"log",'a') as f :
    f.write(f"PREDICTION DONE:{(time.time()-t0)//60}min\n")
