from nn_lib.mdl import CELoss
from nn_lib.optim import SGD
from nn_lib.data import Dataloader

from toy_mlp.model_trainer import ModelTrainer

from mnist.mnist_mlp_classifier import MnistMLPClassifier
from mnist.mnist_dataset import MnistDataset


def main(n_epochs, hidden_layer_sizes, lr=1e-2):
    # create binary MLP classification model
    mlp_model = MnistMLPClassifier(in_features=28*28,out_features=10, hidden_layer_sizes=hidden_layer_sizes)
    print(f'Created the following mnist MLP classifier:\n{mlp_model}')
    # create loss function
    loss_fn = CELoss()
    # create optimizer for model parameters
    optimizer = SGD(mlp_model.parameters(), lr=lr, weight_decay=5e-4, dynamic_lr=True)

    # create a model trainer
    model_trainer = ModelTrainer(mlp_model, loss_fn, optimizer)

    # generate a training dataset
    train_dataset = MnistDataset(train=True)
    # generate a validation dataset different from the training dataset
    val_dataset = MnistDataset(train=False)
    # create a dataloader for training data with shuffling and dropping last batch
    train_dataloader = Dataloader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
    # create a dataloader for validation dataset without shuffling or last batch dropping
    val_dataloader = Dataloader(val_dataset, batch_size=100, shuffle=False, drop_last=False)

    # train the model for a given number of epochs
    model_trainer.train(train_dataloader, n_epochs=n_epochs)

    # validate model on the train data
    # Note: we create a new dataloader without shuffling or last batch dropping
    train_predictions, train_accuracy, train_mean_loss = model_trainer.validate(
        Dataloader(train_dataset, batch_size=100, shuffle=False, drop_last=False))
    print(f'Train accuracy: {train_accuracy:.4f}')
    print(f'Train loss: {train_mean_loss:.4f}')

    # validate model on the validation data
    val_predictions, val_accuracy, val_mean_loss = model_trainer.validate(val_dataloader)
    print(f'Validation accuracy: {val_accuracy:.4f}')
    print(f'Validation loss: {val_mean_loss:.4f}')

    return train_dataset, train_predictions
    # visualize dataset together with its predictions
    #val_dataset.visualize(val_predictions)



def main_dynamic_lr(n_epochs, hidden_layer_sizes, lr=1e-2):
    # create binary MLP classification model
    mlp_model = MnistMLPClassifier(in_features=28*28,out_features=10, hidden_layer_sizes=hidden_layer_sizes)
    print(f'Created the following mnist MLP classifier:\n{mlp_model}')
    # create loss function
    loss_fn = CELoss()
    # create optimizer for model parameters
    optimizer = SGD(mlp_model.parameters(), lr=lr, weight_decay=5e-4)

    # create a model trainer
    model_trainer = ModelTrainer(mlp_model, loss_fn, optimizer)

    # generate a training dataset
    train_dataset = MnistDataset(train=True)
    # generate a validation dataset different from the training dataset
    val_dataset = MnistDataset(train=False)
    # create a dataloader for training data with shuffling and dropping last batch
    train_dataloader = Dataloader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
    # create a dataloader for validation dataset without shuffling or last batch dropping
    val_dataloader = Dataloader(val_dataset, batch_size=100, shuffle=False, drop_last=False)

    # train the model for a given number of epochs
    model_trainer.train(train_dataloader, n_epochs=n_epochs)

    # validate model on the train data
    # Note: we create a new dataloader without shuffling or last batch dropping
    train_predictions, train_accuracy, train_mean_loss = model_trainer.validate(
        Dataloader(train_dataset, batch_size=100, shuffle=False, drop_last=False))
    print(f'Train accuracy: {train_accuracy:.4f}')
    print(f'Train loss: {train_mean_loss:.4f}')

    # validate model on the validation data
    val_predictions, val_accuracy, val_mean_loss = model_trainer.validate(val_dataloader)
    print(f'Validation accuracy: {val_accuracy:.4f}')
    print(f'Validation loss: {val_mean_loss:.4f}')

    return train_dataset, train_predictions
    # visualize dataset together with its predictions
    #val_dataset.visualize(val_predictions)

def mnist_visualise(idx_from = 0, idx_to = 10000):
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.datasets import mnist
    X_test, y_test = mnist.load_data()[1]
    X_test = X_test[idx_from:idx_to]
    y_test = y_test[idx_from:idx_to]
    fig, ax = plt.subplots(
        nrows=2,
        ncols=5,
        sharex=True,
        sharey=True, 
        num=f'first data from {idx_from} to {idx_to} images')
    
    ax = ax.flatten()
    for i in range(10):
        img = X_test[y_test == i][0]
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # mnist_visualise(5000,6000)
    main(n_epochs=5, hidden_layer_sizes=(128,))
