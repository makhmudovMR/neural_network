import numpy as np
from mnist import MNIST

class NeuralNet():
  def __init__(self, input_neuron, hidden_neurons, output_neurons):
    self.w1 = np.random.rand(hidden_neurons, input_neuron)
    self.w2 = np.random.rand(output_neurons, hidden_neurons)

  def activation(self, x):
    return 1 / (1 + np.exp(-x))

  def forward(self, inputs):
    z1 = np.dot(self.w1, inputs)
    o1 = self.activation(z1)
    z2 = np.dot(self.w2, o1)
    o2 = self.activation(z2)
    return o2

  def train(self, inputs, target):
    z1 = np.dot(self.w1, inputs)
    o1 = self.activation(z1)
    z2 = np.dot(self.w2, o1)
    o2 = self.activation(z2)

    d_l2 = (target - o2) * o2 * (1 - o2)
    dw2 = np.dot(d_l2, o1.T)

    d_l1 = np.dot(self.w2.T, (target - o2) * o2 * (1 - o2))
    dw1 = np.dot(d_l1, inputs.T)

    self.w2 = self.w2 + (0.1 * dw2)
    self.w1 = self.w1 + (0.1 * dw1)


def example_train():
  inputs = np.array([
    [1],
    [1],
    [1]
  ])

  target = np.array([
    [0],
    [1]
  ])
  nn = NeuralNet(3,2,2)
  print('before:', nn.forward(inputs))

  for i in range(500):
    nn.train(inputs, target)

  print('after:', nn.forward(inputs))


def train():
  mnist = MNIST()
  nn = NeuralNet(784, 20, 10)
  print(nn.forward(mnist.train_set.images[2].reshape(784, 1)))
  for epoch in range(5):
    for image_index in range(5000):
      image = mnist.train_set.images[image_index]
      label = mnist.train_set.labels[image_index]

      nn.train(image.reshape(784, 1), label.reshape(10, 1))
  print('-----------')
  print(nn.forward(mnist.train_set.images[2].reshape(784, 1)))
  print(mnist.train_set.labels[2])



train()