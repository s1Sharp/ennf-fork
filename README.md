# Educational Neural Network Framework

The purpose of this repository is to serve as a practical material for teaching
students the fundamentals of neural network structure and design.

## Main components

At the moment there are two main components to the repository:

### `nn_lib` package
Contains  an  implementation of a basic neural network library supporting both
forward and backward propagation. The library is inspired by PyTorch -- a popular 
ML framework and can be treated as a very simplified version of it. All operations
are essentially performed on NumPy arrays.

For education purposes some methods implementations are removed and students are
tasked to implement those methods themselves. This way the package is only a template
of an ML framework. Implementing the missing logic should be a valuable exersice for
the students. On the other hand, the logic that is kept should ease the burden of
implementing everything by themselves and focus students only on the core components
responsible for neural network inference and training. 

* `nn_lib.math_fns` implements the expected behaviour of every supported mathematical 
function during both forward (value) and backward (gradient) passes
* `nn_lib.tests` contains rich test base target at checking the correctness of
students' implementations
* `nn_lib.tensor` is the core component of `nn_lib`, implements application of
math operations on Tensors, and gradient propagation and accumulation
* `nn_lib.mdl` contains an interface of a Module class (similar to `torch.nn.Module`)
and some implementations of it
* `nn_lib.optim` contains an interface for an NN optimizer and a Stochastic Gradient
Descent (SGD) optimizer as the simplest version of it
* `nn_lib.data` contains data processing -related components such as Dataset or 
Dataloader

### `toy_mlp` package
An example usage of `nn_lib` package for the purpose of training a small Multi-Layer
Perceptron (MLP) neural network on a toy dataset of 2D points for binary
classification task. Again some methods implementations are removed to be implemented
by students as an exercise.

The example describes a binary MLP NN model (`toy_mlp.binary_mlp_classifier`), 
a synthetically generated 2D toy dataset (`toy_mlp.toy_dataset`), a class for
training and validating a model (`toy_mlp.model_trainer`) and the main execution
script (`toy_mlp.train_toy_mlp`) that demonstrates a regular pipeline of solving a
task using machine learning approach.
