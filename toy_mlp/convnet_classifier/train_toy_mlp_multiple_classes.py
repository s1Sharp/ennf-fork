from nn_lib.mdl.loss_functions import CELoss
from nn_lib.optim import Adam, Optimizer
from nn_lib.data import Dataloader

from toy_mlp.model_trainer import ModelTrainer
from toy_mlp.convnet_classifier.multiple_mlp_classifier import MLPClassifier
from toy_mlp.mnist_classifier.mnist_dataset import MnistDataset
from toy_mlp.convnet_classifier.cifar10_dataset import CIFAR10
from toy_mlp.history_plotter import plot_loss
from nn_lib.mdl.layers import LinearL, Relu, Conv2d,MaxPool2d, Reshape

from nn_lib.scheduler.multi_step_lr import MultiStepLR


def main(n_epochs, layers, optim: Optimizer = Adam, milestones=[], visualize=False):
    # create binary MLP classification model
    mlp_model = MLPClassifier(layers=layers)
    print(f'Created the following binary MLP classifier:\n{mlp_model}')
    # create loss function
    loss_fn = CELoss()
    # create optimizer for model parameters
    optimizer = optim(mlp_model.parameters(), lr=1e-2, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    # create a model trainer
    model_trainer = ModelTrainer(mlp_model, loss_fn, optimizer, scheduler=scheduler)

    # generate a validation dataset different from the training dataset
    #val_dataset = CIFAR10(ds_type='val',gray_scale=True)
    val_dataset = MnistDataset(ds_type='val',conv=True)
    # generate a training dataset
    #train_dataset = CIFAR10(ds_type='train',gray_scale=True)
    train_dataset = MnistDataset(ds_type='train',conv=True)
    # create a dataloader for training data with shuffling and dropping last batch
    train_dataloader = Dataloader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
    # create a dataloader for validation dataset without shuffling or last batch dropping
    val_dataloader = Dataloader(val_dataset, batch_size=100, shuffle=False, drop_last=False)

    # train the model for a given number of epochs
    model_trainer.train(train_dataloader, n_epochs=n_epochs)

    # validate model on the train data
    # Note: we create a new dataloader without shuffling or last batch dropping
    train_predictions, train_accuracy, train_mean_loss = model_trainer.validate(
        Dataloader(train_dataset, batch_size=100, shuffle=False, drop_last=False), multiclass=True)
    print(f'Train accuracy: {train_accuracy:.4f}')
    print(f'Train loss: {train_mean_loss.item():.4f}')

    # validate model on the validation data
    val_predictions, val_accuracy, val_mean_loss = model_trainer.validate(val_dataloader, multiclass=True)
    print(f'Validation accuracy: {val_accuracy:.4f}')
    print(f'Validation loss: {val_mean_loss.item():.4f}')

    # visualize dataset together with its predictions
    if visualize:
        val_dataset.visualize(val_predictions,show_positive=False,number=2)

    # plot_loss(model_trainer.history_loss)
    return model_trainer.history_loss


if __name__ == '__main__':
        main(n_epochs=2,layers=[Conv2d(in_dim=1,out_dim=1,kernel_size=3,stride=1,padding=0),
                              #  Relu(),
                            MaxPool2d(),
                            Reshape([169]),
                            LinearL(in_dim=169, out_dim=10)
                            ],
                            optim=Adam,
                    milestones=[10], visualize=True)