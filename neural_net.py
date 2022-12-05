import numpy as np
from mnist import MNIST

class NeuralNet():
  def __init__(self):
    self.w1 = np.random.rand(2,3)
    self.w2 = np.random.rand(1,2)

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
    dw2 = np.dot(d_l2, o2)

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

  target = np.array([[0]])
  nn = NeuralNet()
  print(nn.forward(inputs))

  for i in range(500):
    nn.train(inputs, target)

  print(nn.forward(inputs))


def train():
  # todo
  pass
