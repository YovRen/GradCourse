import torch
from tqdm import tqdm
from torch_sparse import SparseTensor
from model import Net
from util import plotter

def load_data(prefix):
    name_idx, label_idx = {}, {}
    with open(prefix+".content") as file:
        lines = file.readlines()
        fea_matrix = torch.zeros((len(lines), len(lines[0].strip().split())-2),dtype=torch.float)
        label = torch.zeros((len(lines)),dtype=torch.long)
        for line in tqdm(lines, desc='Processing .content', unit='line'):
            data_split = line.strip().split()
            if data_split[0] not in name_idx:
                name_idx[data_split[0]] = len(name_idx)
            if data_split[-1] not in label_idx:
                label_idx[data_split[-1]] = len(label_idx)
            transform_idx = name_idx[data_split[0]]
            label[transform_idx] = label_idx[data_split[-1]]
            for j in range(len(data_split) - 2):
                fea_matrix[transform_idx][j] = int(data_split[j + 1])
        fea_matrix = fea_matrix / fea_matrix.sum(1, keepdims=True)

    # 创建稀疏矩阵
    with open(prefix + ".cites") as file:
        row_indices = []
        col_indices = []
        for line in tqdm(file, desc='Processing .cites', unit='line'):
            data_split = line.strip().split()
            if data_split[0] in name_idx and data_split[1] in name_idx:
                first_node, second_node = name_idx[data_split[0]], name_idx[data_split[1]]
                row_indices.append(first_node)
                col_indices.append(second_node)
    row_indices = torch.tensor(row_indices)
    col_indices = torch.tensor(col_indices)
    values = torch.ones_like(row_indices, dtype=torch.float)  # 使用 1 表示连接
    adj_matrix = SparseTensor(row=row_indices, col=col_indices, value=values, sparse_sizes=(len(name_idx), len(name_idx)))
    return fea_matrix, adj_matrix, label

fea_matrix, adj_matrix, label = load_data("data/citeseer/citeseer")
indices = torch.randperm(fea_matrix.size(0))
train_size = int(0.8 * fea_matrix.size(0))
valid_size = int(0.1 * fea_matrix.size(0))
train_idx, valid_idx, test_idx = torch.split(indices, [train_size, valid_size, fea_matrix.size(0) - train_size - valid_size])
train_mask = torch.zeros(fea_matrix.shape[0], dtype=torch.bool)
valid_mask = torch.zeros(fea_matrix.shape[0], dtype=torch.bool)
test_mask = torch.zeros(fea_matrix.shape[0], dtype=torch.bool)
train_mask[train_idx], valid_mask[valid_idx], test_mask[test_idx]=True, True, True

def Trainloop(lr, num_layers, alpha, add_self_loops, normalize, dropout, lrd):
    device = torch.device('cpu')
    model = Net(nfeat = fea_matrix.shape[1], nhid=16,nout= len(set(label.tolist())), num_layers=num_layers, alpha=alpha, add_self_loops=add_self_loops, normalize=normalize, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        fea_matrix.to(device)
        adj_matrix.to(device)
        label.to(device)
        predict = model(fea_matrix, adj_matrix)
        train_predict = predict[train_mask]
        train_label = label[train_mask]
        train_loss = loss_fn(train_predict, train_label)
        train_acc = torch.eq(train_predict.max(1).indices, train_label).float().mean(0)
        train_loss.backward()
        optimizer.step()
        if epoch%10==0:
            model.eval()
            with torch.no_grad():
                valid_predict = predict[valid_mask]
                valid_label = label[valid_mask]
                valid_loss = loss_fn(valid_predict, valid_label)
                if lrd:
                    scheduler.step(valid_loss)
                valid_acc = torch.eq(valid_predict.max(1).indices, valid_label).float().mean(0)
            print("Epochs: {:3d} train_loss {:3f} train_acc {:3f} valid_loss {:3f} valid_acc {:3f}".format(epoch, train_loss.item(), train_acc, train_loss.item(), valid_acc))
            train_loss_list.append(train_loss.item())
            train_acc_list.append(train_acc) 
            valid_loss_list.append(valid_loss.item())
            valid_acc_list.append(valid_acc)
    model.eval()
    with torch.no_grad():
        fea_matrix.to(device)
        adj_matrix.to(device)
        label.to(device)
        predict = model(fea_matrix, adj_matrix)
        test_predict = predict[test_mask]
        test_label = label[test_mask]
        test_loss = loss_fn(test_predict, test_label)
        test_acc = torch.eq(test_predict.max(1).indices, test_label).float().mean(0)
    print("test_loss {:3f} test_acc {:3f}".format(test_loss, test_acc))
    curves = [train_loss_list,train_acc_list,valid_loss_list,valid_acc_list]
    values = [test_loss, test_acc]
    return curves,values
# p11 = Trainloop(lr=0.01, num_layers=2, alpha=0.5, add_self_loops=True, normalize=True, dropout=0.2, lrd=False)
# p12 = Trainloop(lr=0.01, num_layers=4, alpha=0.5, add_self_loops=True, normalize=True, dropout=0.2, lrd=False)
# p13 = Trainloop(lr=0.01, num_layers=3, alpha=0.5, add_self_loops=True, normalize=True, dropout=0.2, lrd=False)
# p14 = Trainloop(lr=0.001, num_layers=3, alpha=0.5, add_self_loops=True, normalize=True, dropout=0.2, lrd=False)
# p15 = Trainloop(lr=0.01, num_layers=3, alpha=0, add_self_loops=True, normalize=True, dropout=0.2, lrd=False)
# p16 = Trainloop(lr=0.0001, num_layers=3, alpha=0.5, add_self_loops=True, normalize=True, dropout=0.2, lrd=False)
# p17 = Trainloop(lr=0.01, num_layers=3, alpha=0.5, add_self_loops=True, normalize=False, dropout=0.2, lrd=False)
# p18 = Trainloop(lr=0.01, num_layers=3, alpha=0.5, add_self_loops=True, normalize=True, dropout=0, lrd=False)
# p19 = Trainloop(lr=0.01, num_layers=3, alpha=0.5, add_self_loops=True, normalize=True, dropout=0.2, lrd=True)
# p10 = Trainloop(lr=0.01, num_layers=3, alpha=0.5, add_self_loops=False, normalize=True, dropout=0.2, lrd=False)
# plotter("citeseer",['dept-1','dept+1','std','lr/10','no_res','lr/100','no_norm','no_drop','lrd','no_self_loops'],[p11,p12,p13,p14,p15,p16,p17,p18,p19,p10])


p11 = Trainloop(lr=0.1, num_layers=2, alpha=0.5, add_self_loops=True, normalize=True, dropout=0.2, lrd=False)
plotter("siteseer2",['net2'],[p11])