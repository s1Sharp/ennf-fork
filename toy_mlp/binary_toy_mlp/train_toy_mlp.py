from nn_lib.mdl.loss_functions import BCELoss
from nn_lib.optim import SGD, Adam, Optimizer
from nn_lib.data import Dataloader

from toy_mlp.model_trainer import ModelTrainer
from toy_mlp.binary_toy_mlp.binary_mlp_classifier import BinaryMLPClassifier
from toy_mlp.binary_toy_mlp.toy_dataset_binary import ToyDatasetBinary
from toy_mlp.history_plotter import plot_loss

from nn_lib.scheduler.multi_step_lr import MultiStepLR
from nn_lib.scheduler.scheduler import Scheduler


def main(n_samples, structure, n_epochs, hidden_layer_sizes,optim:Optimizer=SGD,milesotnes=[], visualize=False):
    # create binary MLP classification model
    mlp_model = BinaryMLPClassifier(in_features=2, hidden_layer_sizes=hidden_layer_sizes)
    print(f'Created the following binary MLP classifier:\n{mlp_model}')
    # create loss function
    loss_fn = BCELoss()
    # create optimizer for model parameters
    optimizer = optim(mlp_model.parameters(), lr=1e-2, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer,milestones=milesotnes,gamma=0.1,last_epoch=100)

    # create a model trainer
    model_trainer = ModelTrainer(mlp_model, loss_fn, optimizer,scheduler=scheduler)

    # generate a training dataset
    train_dataset = ToyDatasetBinary(n_samples=n_samples, structure=structure, seed=0)
    # generate a validation dataset different from the training dataset
    val_dataset = ToyDatasetBinary(n_samples=n_samples, structure=structure, seed=1)
    # create a dataloader for training data with shuffling and dropping last batch
    train_dataloader = Dataloader(train_dataset, batch_size=90, shuffle=True, drop_last=True)
    # create a dataloader for validation dataset without shuffling or last batch dropping
    val_dataloader = Dataloader(val_dataset, batch_size=90, shuffle=False, drop_last=False)

    # train the model for a given number of epochs
    model_trainer.train(train_dataloader, n_epochs=n_epochs)

    # validate model on the train data
    # Note: we create a new dataloader without shuffling or last batch dropping
    train_predictions, train_accuracy, train_mean_loss = model_trainer.validate(
        Dataloader(train_dataset, batch_size=90, shuffle=False, drop_last=False))
    print(f'Train accuracy: {train_accuracy:.4f}')
    print(f'Train loss: {train_mean_loss:.4f}')

    # validate model on the validation data
    val_predictions, val_accuracy, val_mean_loss = model_trainer.validate(val_dataloader)
    print(f'Validation accuracy: {val_accuracy:.4f}')
    print(f'Validation loss: {val_mean_loss:.4f}')

    # visualize dataset together with its predictions
    if visualize:
        val_dataset.visualize(val_predictions)

    #plot_loss(model_trainer.history_loss)
    return model_trainer.history_loss

if __name__ == '__main__':

    main(n_samples=1000, structure='blobs', n_epochs=100, hidden_layer_sizes=(20,),visualize=True)
    #plot_loss([main(n_samples=1000, structure='moons', n_epochs=100, hidden_layer_sizes=(400,),optim=Adam,visualize=True),
     #           main(n_samples=1000, structure='moons', n_epochs=100, hidden_layer_sizes=(400,),optim=Adam,milesotnes=[70],visualize=True)
      #         ], ['m1','m2'])
    #plot_loss([main(n_samples=1000, structure='moons', n_epochs=100, hidden_layer_sizes=(500,),visualize=True)])
    #plot_loss([main(n_samples=1000, structure='moons', n_epochs=100, hidden_layer_sizes=(100,), optim=SGD,visualize=True),
     #         main(n_samples=1000, structure='moons', n_epochs=100, hidden_layer_sizes=(100,), optim=Adam, visualize=True)],
      #        ['SGD','Adam'])

    #plot_loss(
       # [main(n_samples=1000, structure='moons', n_epochs=100, hidden_layer_sizes=(100,), optim=SGD, visualize=True),
       #  main(n_samples=1000, structure='moons', n_epochs=100, hidden_layer_sizes=(100,), optim=Adam, visualize=True)],
       # ['SGD', 'Adam'])
