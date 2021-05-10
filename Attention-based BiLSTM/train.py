# -*- coding: utf-8 -*-
import numpy as np
import pickle
import sys
import codecs
import importlib
importlib.reload(sys)


with open('./data/train2.pkl', 'rb') as inp:
    word2id = pickle.load(inp, encoding='iso-8859-1')
    id2word = pickle.load(inp, encoding='iso-8859-1')
    relation2id = pickle.load(inp, encoding='iso-8859-1')
    train = pickle.load(inp, encoding='iso-8859-1')
    labels = pickle.load(inp, encoding='iso-8859-1')
    position1 = pickle.load(inp, encoding='iso-8859-1')
    position2 = pickle.load(inp, encoding='iso-8859-1')

# with open('./data/engdata_test.pkl', 'rb') as inp:
with open('./data/test2.pkl', 'rb') as inp:
    test = pickle.load(inp, encoding='iso-8859-1')
    labels_t = pickle.load(inp, encoding='iso-8859-1')
    position1_t = pickle.load(inp, encoding='iso-8859-1')
    position2_t = pickle.load(inp, encoding='iso-8859-1')

    

   
print("train len", len(train))
print("test len", len(test))
print("word2id len",len(word2id))

import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.data as D
from torch.autograd import Variable
from BiLSTM_ATT import BiLSTM_ATT


EMBEDDING_SIZE = len(word2id)+1   # +1 ?
EMBEDDING_DIM = 100

POS_SIZE = 82  #不同数据集这里可能会报错。
POS_DIM = 25

HIDDEN_DIM = 200

TAG_SIZE = len(relation2id)
# print(len(relation2id))
# TAG_SIZE = 54
BATCH = 128
EPOCHS = 100

config={}
config['EMBEDDING_SIZE'] = EMBEDDING_SIZE
config['EMBEDDING_DIM'] = EMBEDDING_DIM
config['POS_SIZE'] = POS_SIZE
config['POS_DIM'] = POS_DIM
config['HIDDEN_DIM'] = HIDDEN_DIM
config['TAG_SIZE'] = TAG_SIZE
config['BATCH'] = BATCH
config["pretrained"]=False

learning_rate = 0.0005


embedding_pre = []
if len(sys.argv)==2 and sys.argv[1]=="pretrained":
    print("use pretrained embedding")
    config["pretrained"]=True
    word2vec = {}
    with codecs.open('vec.txt','r','utf-8') as input_data:
        for line in input_data.readlines():
            word2vec[line.split()[0]] = map(eval,line.split()[1:])

    unknow_pre = []
    unknow_pre.extend([1]*100)
    embedding_pre.append(unknow_pre) #wordvec id 0
    for word in word2id:
        if word in word2vec:
            embedding_pre.append(word2vec[word])
        else:
            embedding_pre.append(unknow_pre)

    embedding_pre = np.asarray(embedding_pre)
    print(embedding_pre.shape)

model = BiLSTM_ATT(config,embedding_pre)
#model = torch.load('model/model_epoch20.pkl')
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(size_average=True)



train = torch.LongTensor(train[:len(train)-len(train)%BATCH])
position1 = torch.LongTensor(position1[:len(train)-len(train)%BATCH])
position2 = torch.LongTensor(position2[:len(train)-len(train)%BATCH])
labels = torch.LongTensor(labels[:len(train)-len(train)%BATCH])
train_datasets = D.TensorDataset(train,position1,position2,labels)
train_dataloader = D.DataLoader(train_datasets,BATCH,True,num_workers=0)


test = torch.LongTensor(test[:len(test)-len(test)%BATCH])
position1_t = torch.LongTensor(position1_t[:len(test)-len(test)%BATCH])
position2_t = torch.LongTensor(position2_t[:len(test)-len(test)%BATCH])
labels_t = torch.LongTensor(labels_t[:len(test)-len(test)%BATCH])
test_datasets = D.TensorDataset(test,position1_t,position2_t,labels_t)
test_dataloader = D.DataLoader(test_datasets,BATCH,True,num_workers=0)


for epoch in range(EPOCHS):
    print("epoch:",epoch)
    acc=0
    total=0

    for sentence,pos1,pos2,tag in train_dataloader:
        sentence = Variable(sentence)
        pos1 = Variable(pos1)
        pos2 = Variable(pos2)
        y = model(sentence,pos1,pos2)
        tags = Variable(tag)
        loss = criterion(y, tags)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y = np.argmax(y.data.numpy(),axis=1)

        for y1,y2 in zip(y,tag):
            if y1==y2:
                acc+=1
            total+=1

    print("train:",100*float(acc)/total,"%")

    with open("accuracy.txt", "a+", encoding='utf-8') as f:
        f.write("epoch: "+str(epoch)+"\n")
        accuracy_num = 100*float(acc)/total
        f.write("train:"+str(accuracy_num)+"%" + "\n")

    acc_t=0
    total_t=0
    count_predict = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    count_total = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    count_right = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for sentence,pos1,pos2,tag in test_dataloader:
        sentence = Variable(sentence)
        pos1 = Variable(pos1)
        pos2 = Variable(pos2)
        y = model(sentence,pos1,pos2)
        y = np.argmax(y.data.numpy(),axis=1)
        for y1,y2 in zip(y,tag):
            count_predict[y1]+=1
            count_total[y2]+=1
            if y1==y2:
                count_right[y1]+=1


    precision = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    recall = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(len(count_predict)):
        if count_predict[i]!=0 :
            precision[i] = float(count_right[i])/count_predict[i]
            print("Precision for relation type "+str(i)+": "+str(precision[i]))
            with open("accuracy.txt", "a+", encoding='utf-8') as f1:
                f1.write("Precision for relation type "+str(i)+": "+str(precision[i]) + "\n")

        if count_total[i]!=0:
            recall[i] = float(count_right[i])/count_total[i]
            print("Recall for relation type " + str(i) + ": " + str(recall[i]))
            with open("accuracy.txt", "a+", encoding='utf-8') as f2:
                f2.write("Recall for relation type " + str(i) + ": " + str(recall[i]) + "\n")


    precision = sum(precision)/len(relation2id)
    recall = sum(recall)/len(relation2id)
    print("Average Precision：",precision)
    print("Average Recall：",recall)
    print("Average F1：", (2*precision*recall)/(precision+recall))

    with open("accuracy.txt", "a+", encoding='utf-8') as f3:
        f3.write("Average Precision："+str(precision) + "\n")
        f3.write("Average Recall："+str(recall) + "\n")
        f1_num = (2*precision*recall)/(precision+recall)
        f3.write("Average F1："+ str(f1_num) + "\n")


    if epoch%20==0:
        model_name = "./model/model_epoch"+str(epoch)+".pkl"
        torch.save(model, model_name)
        print(model_name,"has been saved")


torch.save(model, "./model/model_01.pkl")
print("model has been saved")


