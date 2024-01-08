from nn_lib.mdl.loss_functions import CELoss
from nn_lib.optim import SGD, Adam, Optimizer
from nn_lib.data import Dataloader

from toy_mlp.model_trainer import ModelTrainer
from toy_mlp.multiple_toy_mlp.multiple_mlp_classifier import MultipleMLPClassifier
from toy_mlp.mnist_classifier.mnist_dataset import MnistDataset
from toy_mlp.history_plotter import plot_loss

from nn_lib.scheduler.multi_step_lr import MultiStepLR


def main(n_epochs, hidden_layer_sizes, optim: Optimizer = Adam, milestones=[], visualize=False):
    # create binary MLP classification model
    mlp_model = MultipleMLPClassifier(in_features=784, hidden_layer_sizes=hidden_layer_sizes)
    print(f'Created the following binary MLP classifier:\n{mlp_model}')
    # create loss function
    loss_fn = CELoss()
    # create optimizer for model parameters
    optimizer = optim(mlp_model.parameters(), lr=1e-2, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    # create a model trainer
    model_trainer = ModelTrainer(mlp_model, loss_fn, optimizer, scheduler=scheduler)

    # generate a training dataset
    train_dataset = MnistDataset(ds_type='train')
    # generate a validation dataset different from the training dataset
    val_dataset = MnistDataset(ds_type='val')
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
        val_dataset.visualize(val_predictions,show_positive=True)

    # plot_loss(model_trainer.history_loss)
    return model_trainer.history_loss


if __name__ == '__main__':
    # main(n_samples=1000, structure='blobs', n_epochs=100, hidden_layer_sizes=(20,))
    # main(n_samples=1000, structure='circles', n_epochs=100, hidden_layer_sizes=(100,),visualize=True)
    # main(n_samples=1000, structure='moons', n_epochs=100, hidden_layer_sizes=(10000,),visualize=True)
    plot_loss([main(n_epochs=5, hidden_layer_sizes=(128, 10), optim=Adam,
                    milestones=[500], visualize=True)],
              ['lloos'])