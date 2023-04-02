# XOR_NN
Making a XOR gate using a Nueral Network made from scratch 

<p>Making a XOR Neural network is a toy problem in deep learning which involves training a nueral network on 
data consisting of input and output of a XOR gate.</p>

The dataset comprimises as follows:

X = [[0, 0]<br/>
     [0, 1]<br/>
     [1, 0]<br/>
     [1, 1]]<br/>
     
Y = [[0], <br/>
     [1],<br/>
     [1],<br/>
     [0]]
     
This repository consists of a two layered nueral network made from scratch using the numpy library in python. The 2 layered nueral network is made 
such that the number of inputs and number of perceptrons in the first layer are configurable while the output remains as the activation of a single perceptron.

Taking size of the first layer as 16, learning rate of 1 and number of iterations for gradient descent as 1000, following output was achieved on the
dataset:

Y = [[0.01584553]<br/>
     [0.98070492]<br/>
     [0.98384326]<br/>
     [0.0199659 ]]
     
giving a cross entropy loss of 0.017978.

In comparison, the loss before gradient descent was 3.072160
  
