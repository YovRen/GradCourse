import matplotlib.pyplot as plt

# 读取文件内容
with open('data.txt', 'r') as file:
    lines = file.readlines()

with open('data2.txt', 'r') as file:
    lines2 = file.readlines()

epochs = []
train_auc_roc = []
train_loss = []
train_auc_roc2 = []
train_loss2 = []

# 解析每行的内容
for line in lines:
    if line.startswith("epoch"):
        parts = line.strip().split()
        epoch = int(parts[1])
        auc_roc = float(parts[-1])
        loss = float(parts[-3])
        epochs.append(epoch)
        train_auc_roc.append(auc_roc)
        train_loss.append(loss)

for line in lines2:
    if line.startswith("epoch"):
        parts = line.strip().split()
        auc_roc = float(parts[-1])
        loss = float(parts[-3])
        train_auc_roc2.append(auc_roc)
        train_loss2.append(loss)

# 创建一行两列的布局
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 绘制 AUC-ROC 曲线图
ax1.plot(epochs, train_auc_roc, label='one-stage')
ax1.plot(epochs, train_auc_roc2, label='two-stages')
ax1.set_title('Train AUC-ROC Curve')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('AUC-ROC Score')
ax1.legend()

# 绘制 Loss 曲线图
ax2.plot(epochs, train_loss, label='one-stage')
ax2.plot(epochs, train_loss2, label='two-stages')
ax2.set_title('Train Loss Curve')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

# 调整布局，以防止重叠
plt.tight_layout()

# 显示图形
plt.show()
