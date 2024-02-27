import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np
from nltk.tokenize import word_tokenize
import gensim
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim

def word2index():
    corpus = []
    stopwords=set([word.strip() for word in open('stopwords.txt',encoding='utf-8')])
    with open('yelp_academic_dataset_review.json', 'r', encoding='utf-8') as file:
        for line in file:
            corpus.append([word.lower() for word in word_tokenize(json.loads(line)['text']) if word.lower() not in stopwords])
    model=gensim.models.word2vec.Word2Vec(corpus,size=300,min_count=10)
    word2vec = [[0] * 300] * 4 + [model[word] for word in word2index]
    word2index = {word: i + 4 for i, word in enumerate(model.wv.index2word)}
    json.dump(word2index, open('word2index.json', 'w', encoding='utf-8'), ensure_ascii=False)
    np.save('word2vec.npy', word2vec)
    
class Data(Dataset):
    def __init__(self, texts, stars, args):
        assert len(texts)==len(stars)
        self.pad = args.pad
        self.start = args.start
        self.end = args.end
        self.unk = args.unk
        self.vec_len = args.vec_len
        self.word2index = args.word2index
        self.word2vec = args.word2vec
        self.texts = [word_tokenize(text) for text in texts]
        self.stars = [star for star in stars]
        self.output_size=args.output_size
        
    def text2vec(self, text):
        """
        对文本进行填充和词向量化
        """
        vector = np.empty((0, 300))
        for word in text:
            vector = np.concatenate((vector, self.word2vec[self.word2index.get(word,self.unk)].reshape(1,-1)),axis=0)
        if len(vector)>=self.vec_len:
            return vector[-self.vec_len:]
        else:
            return np.concatenate((vector, np.stack([self.word2vec[self.pad] for _ in range(self.vec_len - len(vector))])),axis=0)

    def __getitem__(self, idx):
        text = self.texts[idx]
        star = self.stars[idx]
        return torch.tensor(self.text2vec(text),dtype=torch.float), torch.eye(self.output_size)[star - 1]
    
    def __len__(self):
        return len(self.texts)

class conv_norm_relu_drop(nn.Module):
    """卷积层
    """
    def __init__(self, in_channel, out_channel, dropout, normalize=False):
        super(conv_norm_relu_drop, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=5, padding=2).to(args.device)
        self.normalize = normalize
        if normalize:
            self.norm = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv(x)
        if self.normalize:
            x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class residual_block(nn.Module):
    """残差网络
    """
    def __init__(self, hidden_size, dropout=False, normalize=False):
        super(residual_block, self).__init__()
        self.conv1 = conv_norm_relu_drop(hidden_size,hidden_size,dropout,normalize)
        self.conv2 = conv_norm_relu_drop(hidden_size,hidden_size,dropout,normalize)
        
    def forward(self, x):
        x_ = self.conv1(x)
        x_ = self.conv2(x_)
        return x + x_

class RNN(nn.Module):
    def __init__(self, args, dropout=0.5, normalization=False, residual=False, num_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = args.hidden_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.batch_size = args.batch_size
        self.conv_size = args.conv_size
        self.rnn_layers = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        self.normalization = normalization
        self.residual = residual
        self.conv1 = conv_norm_relu_drop(self.input_size, self.conv_size ,self.dropout, self.normalization).to(args.device)
        self.res1 = residual_block(self.conv_size ,self.dropout, self.normalization).to(args.device)
        self.conv2 = conv_norm_relu_drop(self.conv_size, self.conv_size ,self.dropout, self.normalization).to(args.device)
        self.res2 = residual_block(self.conv_size ,self.dropout, self.normalization).to(args.device)
        self.rnn = nn.RNN(self.conv_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout).to(args.device)
        self.conv3 = conv_norm_relu_drop(self.hidden_size, self.hidden_size ,self.dropout, self.normalization).to(args.device)
        self.res3 = residual_block(self.hidden_size ,self.dropout, self.normalization).to(args.device)
        self.fc = nn.Linear(self.hidden_size, self.output_size).to(args.device)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv1(x)
        if self.residual:
            x = self.res1(x)
        x = self.conv2(x)
        if self.residual:
            x = self.res2(x)
        hidden = torch.rand(self.num_layers, x.size(0), self.hidden_size).to(args.device)
        x = x.permute(0,2,1)
        output, hidden = self.rnn(x, hidden)
        output = output.permute(0,2,1)
        output = self.conv3(output)
        if self.residual:
            output = self.res3(output)
        output = output.permute(0,2,1)
        output = self.fc(output[:, -1, :])
        return output


def acc(labels, outputs, type_="top1"):
    acc = 0
    if type_ == "top1":
        pre_labels = np.argmax(outputs, axis=1)
        labels = np.argmax(labels, axis=1)
        acc = np.sum(pre_labels == labels) / len(pre_labels)

    return acc

def Training(args, net, trainloader, valloader, lrd, optimizer,loss_func,scheduler):
    print("start training-----------------------")
    epochs = args.epochs
    device = args.device
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    for i in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        
        for idx, (inputs,labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            train_loss += loss.item()
            train_acc += acc(labels=labels.cpu().numpy(), outputs=outputs.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            if (idx+1) % (len(trainloader)//5) == 0:
                train_loss = train_loss / (len(trainloader)//5)
                train_acc = train_acc / (len(trainloader)//5)
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                net.eval()
                for inputs, labels in valloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    loss = loss_func(outputs, labels)
                    val_loss += loss.item()
                    val_acc += acc(labels=labels.cpu().numpy(), outputs=outputs.detach().cpu().numpy())
                val_loss = val_loss / len(valloader)
                val_acc = val_acc / len(valloader)
                val_loss_list.append(val_loss)
                val_acc_list.append(val_acc)
                net.train()
                print(f"Epoch {i}({100*(idx+1)/len(trainloader):3.1f}%): train_loss {train_loss:10.6f}, train_acc {train_acc:7.4f}, val_loss {val_loss:10.6f}, val_acc {val_acc:7.4f}")

                if lrd:
                    scheduler.step(val_loss)
                train_loss = 0.0
                train_acc = 0.0
                val_loss = 0.0
                val_acc = 0.0
    return [train_loss_list, val_loss_list, train_acc_list, val_acc_list]

def plotter(title,p):
    fig, axs = plt.subplots(2, 2, figsize=(16, 9), dpi=60)
    x = range(len(p[0][0]))
    axs[0,0].set_title('loss_train')
    axs[0,1].set_title('val_loss')
    axs[1,0].set_title('train_acc')
    axs[1,1].set_title('val_acc')
    legend = []
    for i in range(len(title)):
        legend.extend([title[i]])
        axs[0,0].plot(x, p[i][0])
        axs[0,1].plot(x, p[i][1])
        axs[1,0].plot(x, p[i][2])
        axs[1,1].plot(x, p[i][3])
    axs[0,0].legend(legend)
    axs[0,1].legend(legend)
    axs[1,0].legend(legend)
    axs[1,1].legend(legend)
    plt.savefig("out.png")
    plt.show()
    with open("out.txt", "w") as file:
        for i in range(len(title)):
            file.write(title[i]+":"+'\t'.join(map(str, np.array(p[i])[:,-1]))+"\n")
            

class Args:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 64
        self.num_workers = 0

        self.input_size = 300
        self.conv_size = 200
        self.hidden_size = 128
        self.output_size = 5
        self.epochs = 3

        self.pad = 0
        self.start = 1
        self.end = 2
        self.unk = 3
        self.vec_len = 200
        self.word2index = json.load(open('word2index.json', encoding='utf-8'))
        self.word2vec = np.load('word2vec.npy')
        self.lr = 0.01


if __name__ == '__main__':
    args = Args()

    texts = []
    stars = []
    with open('small.json', 'r', encoding='utf-8') as file:
        for line in file:
            texts.append(json.loads(line)['text'])
            stars.append(json.loads(line)['stars'])
    train_texts, test_texts, train_stars, test_stars = train_test_split(texts, stars, test_size=0.05, random_state=42)
    train_dataset = Data(train_texts, train_stars, args)
    test_dataset = Data(test_texts, test_stars, args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, drop_last=False)
    
    set_ups = [
    {'lrd':True,'dropout':0,'normalization':True,'residual':True,'num_layers':4},
    {'lrd':False,'dropout':0,'normalization':True,'residual':True, 'num_layers':4},
    {'lrd':True,'dropout':0.5,'normalization':True,'residual':True, 'num_layers':4},
    {'lrd':True,'dropout':0.2,'normalization':True,'residual':True, 'num_layers':4},
    {'lrd':True,'dropout':0,'normalization':False,'residual':True, 'num_layers':4},
    {'lrd':True,'dropout':0,'normalization':True,'residual':False,'num_layers':4},
    {'lrd':True,'dropout':0,'normalization':True,'residual':True,'num_layers':5},
    {'lrd':True,'dropout':0,'normalization':True,'residual':True,'num_layers':3},
    {'lrd':True,'dropout':0,'normalization':True,'residual':True,'num_layers':2}
    ]
    results = []
    for set_up in set_ups:
        rnn = RNN(args,set_up['dropout'],set_up['normalization'],set_up['residual'],set_up['num_layers'])
        optimizer = optim.Adam(rnn.parameters(), lr=args.lr)
        loss_func = nn.MSELoss()
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, verbose=True)
        result = Training(args, net=rnn, trainloader=train_loader, valloader=val_loader, lrd=set_up['lrd'], optimizer=optimizer, loss_func=loss_func, scheduler=scheduler)
        results.append(result)
    plotter(['std','no_lrd','drop_0.5','drop_0.2','no_norm','no_res','depth+1','dept-1','dept-2'],results)