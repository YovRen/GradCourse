import matplotlib.pyplot as plt

def plotter(out_path,title,p):
    fig, axs = plt.subplots(2, 2, figsize=(16, 9), dpi=300)
    x = range(len(p[0][0][0]))
    axs[0,0].set_title('loss_train')
    axs[0,1].set_title('loss_val')
    axs[1,0].set_title('acc_train')
    axs[1,1].set_title('acc_val')
    legend = []
    for i in range(len(title)):
        legend.extend([title[i]])
        target_length = len(p[0][0][0])  # 获取目标长度
        for j in range(4):  # 遍历要处理的位置索引
            current_length = len(p[i][0][j])  # 获取当前长度
            if current_length < target_length:  # 如果长度不足，用最后一个值填充
                p[i][0][j].extend(p[i][0][j][-1] * (target_length - current_length))
        axs[0,0].plot(x, p[i][0][0])
        axs[0,1].plot(x, p[i][0][1])
        axs[1,0].plot(x, p[i][0][2])
        axs[1,1].plot(x, p[i][0][3])
    axs[0,0].legend(legend)
    axs[0,1].legend(legend)
    axs[1,0].legend(legend)
    axs[1,1].legend(legend)
    plt.savefig(out_path+f"out.png")
    plt.show()
    with open(out_path+f"out.txt", "w") as file:
        for i in range(len(title)):
            file.write(title[i]+":"+'\t'.join(map(str, p[i][1]))+"\n")