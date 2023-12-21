from matplotlib import pyplot as plt


def plot_loss(losses=list[list], loss_names: list[str] = ['train loss'], tittle: str = 'Loss over epoches'):
    plt.figure(figsize=(15, 9))
    for loss, ttl in zip(losses, loss_names):
        plt.plot(loss, label=ttl)

    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(tittle)
    plt.show()
