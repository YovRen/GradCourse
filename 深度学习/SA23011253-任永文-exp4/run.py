import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from model import Net2
from util import plotter

pre_transform = T.Compose([T.GCNNorm(), T.ToSparseTensor()])
train_dataset = PPI("data/ppi", split='train')
valid_dataset = PPI("data/ppi", split='val', pre_transform=pre_transform)
test_dataset = PPI("data/ppi", split='test', pre_transform=pre_transform)
train_loader = DataLoader(train_dataset, shuffle=True)
valid_loader = DataLoader(valid_dataset, shuffle=False)
test_loader = DataLoader(test_dataset, shuffle=False)


def Trainloop(lr, num_layers, alpha, add_self_loops, normalize, dropout, lrd):
    device = torch.device('cuda')
    # model = Net(nfeat=train_dataset.num_features, nhid=512, nout=train_dataset.num_classes, num_layers=num_layers, alpha=alpha, add_self_loops=add_self_loops, normalize=normalize, dropout=dropout).to(device)
    model = Net2(nfeat=train_dataset.num_features, nhid=512, nout=train_dataset.num_classes, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    for epoch in range(1000):
        model.train()
        train_loss = train_acc = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.adj_t)
            loss = loss_fn(out, data.y)
            acc = ((torch.sigmoid(out) > 0.5).long() == data.y).all(dim=1).float().mean().item()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += acc
        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_loader)
        model.eval()
        if epoch%10==0:
            valid_loss = valid_acc = 0
            for data in valid_loader:
                data = data.to(device)
                out = model(data.x, data.adj_t)
                loss = loss_fn(out, data.y)
                if lrd:
                    scheduler.step(loss)
                acc = ((torch.sigmoid(out) > 0.5).long() == data.y).all(dim=1).float().mean().item()
                valid_loss += loss.item()
                valid_acc += acc
            valid_loss = valid_loss / len(valid_loader)
            valid_acc = valid_acc / len(valid_loader)
            print("Epochs: {:3d} train_loss {:3f} train_acc {:3f} valid_loss {:3f} valid_acc {:3f}".format(epoch, train_loss, train_acc, train_loss, valid_acc))
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc) 
            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_acc)
    model.eval()
    test_loss = test_acc = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.adj_t)
        loss = loss_fn(out, data.y)
        acc = ((torch.sigmoid(out) > 0.5).long() == data.y).all(dim=1).float().mean().item()
        test_loss += loss.item()
        test_acc += acc
    test_loss = test_loss / len(test_loader)
    test_acc = test_acc / len(test_loader)
    print("test_loss {:3f} test_acc {:3f}".format(test_loss, test_acc))
    curves = [train_loss_list,train_acc_list,valid_loss_list,valid_acc_list]
    values = [test_loss, test_acc]
    return curves,values

# p11 = Trainloop(lr=0.001, num_layers=2, alpha=0.5, add_self_loops=True, normalize=True, dropout=0.2, lrd=False)
# p12 = Trainloop(lr=0.001, num_layers=4, alpha=0.5, add_self_loops=True, normalize=True, dropout=0.2, lrd=False)
# p13 = Trainloop(lr=0.001, num_layers=3, alpha=0.5, add_self_loops=True, normalize=True, dropout=0.2, lrd=False)
# p14 = Trainloop(lr=0.0001, num_layers=3, alpha=0.5, add_self_loops=True, normalize=True, dropout=0.2, lrd=False)
# p15 = Trainloop(lr=0.001, num_layers=3, alpha=0, add_self_loops=True, normalize=True, dropout=0.2, lrd=False)
# p16 = Trainloop(lr=0.01, num_layers=3, alpha=0.5, add_self_loops=True, normalize=True, dropout=0.2, lrd=False)
# p17 = Trainloop(lr=0.001, num_layers=3, alpha=0.5, add_self_loops=True, normalize=False, dropout=0.2, lrd=False)
# p18 = Trainloop(lr=0.001, num_layers=3, alpha=0.5, add_self_loops=True, normalize=True, dropout=0, lrd=False)
# p19 = Trainloop(lr=0.001, num_layers=3, alpha=0.5, add_self_loops=True, normalize=True, dropout=0.2, lrd=True)
# p10 = Trainloop(lr=0.001, num_layers=3, alpha=0.5, add_self_loops=False, normalize=True, dropout=0.2, lrd=False)
# plotter("ppi",['dept-1','dept+1','std','lr/10','no_res','lr*10','no_norm','no_drop','lrd','no_self_loops'],[p11,p12,p13,p14,p15,p16,p17,p18,p19,p10])

p11 = Trainloop(lr=0.001, num_layers=2, alpha=0.5, add_self_loops=True, normalize=True, dropout=0.2, lrd=False)
plotter("ppi2",['dept-1'],[p11])