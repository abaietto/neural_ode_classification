# ECG Classification

## Problem Statement
As the field of deep learning matures, better and faster neural network architectures are likely to be researched and deployed in industry. In this project, we investigate one such state-of-the-art architecture called Neural ODEs. We compare it to its predecessor, the residual neural network or ResNet. Differences in training/testing time, memory, and accuracy will be noted in the context of a supervised learning task, namely the classification of an ECG signal.

## Executive Summary
We implement both the Nueral ODE and ResNet architectures for classification of electrocardiogram signals. The data is taken from the frequently used MIT-BIH ECG database, which contains over 100,000 labeled samples of ECG signals from a single heartbeat. The data and a description of the sampling procedure used can be found at https://www.physionet.org/content/mitdb/1.0.0/. We briefly visualize the data, which enables us an intuitive look at the various features and differences between each class that our neural networks will learn and distinguish. 

We construct both network architectures with the help of PyTorch and the torchdiffeq library found in the original Neural ODE paper: https://arxiv.org/abs/1806.07366. The ResNet and ODENet, the implementation of the Neural ODE, are designed to be as similar as possible, so we can compare the two fairly. Both models contain identical downampling layers, 1D convolutions, normalizations (Group), activations (ReLU), and output layers. We train both models and evaluate them on the testing set while also noting differences in speed, memory, and accuracy. Overall, both models perform comparably on the testing data. However, there is a tradeoff between speed and memory. The ResNet is faster to train while the ODENet contains fewer tunable parameters (less memory).

### Contents:
- [Background](#Background)
- [EDA](#EDA)
- [Network Building](#Network-Building)
- [Modeling](#Modeling)
- [Conclusion](#Conclusion)

## Background

Two neural network architectures that have generated a lot of buzz in deep learning communities are residual neural networks and neural ordinary differential equations, also known as ResNets and Neural ODEs, respectively. The ResNet came first in 2015 with the paper https://arxiv.org/abs/1512.03385 and was used to solve a significant problem in deep learning. As more and more layers were added to a typical network, the accuracy would decrease! In order to combat this, shortcut connections were included in the network. This means that the input of one layer is directly added to the transformed input and the sum of these components comprises the layer's output. This is called a residual block and is the basic unit that makes up a ResNet. Typically, tens or hundreds of these blocks are linked together to build a full ResNet.

![ResBlock](resblock.png)

As an equation a ResNet looks like h<sub>t+1</sub> = h<sub>t</sub> + f(h<sub>t</sub>, &theta;<sub>t</sub>). The input of a hidden layer, h<sub>t</sub>, is added to a function of that layer and the model parameters in order to get the next hidden layer, h<sub>t+1</sub>. These functions are the usual neural network transformations, such as convolutions, normalizations, and linear mappings. All in all, the ResNet ended up being very successful and performed better than ever before on numerous image classification tasks.

A couple years later in 2018, Neural ODEs were born. The authors of the original paper noticed that the ResNet equation looks an awful lot like Euler's method from DiffEq-101. Euler's method is a way to numerically approximate the solution to a differential equation. It involves adding an initial condition to the product of a stepsize and the slope at that point. For a given differential equation, dy/dt = f(t, y(t)), Euler's method yields the solution y<sub>n+1</sub> = y<sub>n</sub> + h * f(t<sub>n</sub>, y<sub>n</sub>) where h is the step size. The authors of Neural ODEs did the opposite and went from a discrete "approximation", the ResNet, to the continuous version, neural ordinary differential equations.

Neural ODEs are governed by the differential equation dh(t)/dt = f(h(t), t, &theta;). Starting from an initial condition at time 0 (the input to the neural network), the solution of the equation is the output. We can rely on hundreds of years of differential equations research to help us find this answer. Training is done by calculating the output for a given output using one of many numerical diffeq solvers and then backpropagating using a clever trick: the adjoint method. More information on the adjoint method can be found in the paper https://arxiv.org/pdf/1806.07366.pdf. There are few important aspects of neural ODEs. For one, they model continuous dynamics naturally. This leads to continuous normalizing flows, which is beyond the scope of this project. Second, they use constant memory due to the nature of the adjoint method and the model itself. This means as more and more "layers" are added, the memory footprint does not increase unlike typical neural networks (including ResNets). Finally, approximation errors can be altered. Depending on one's needs, one can balance speed versus accuracy. This is achieved by altering the tolerances/errors of the diffeq solvers. An interesting option is to train with small error to maximize accuracy and then speed up modeling for production by increasing the error. Neural ODEs offer many advantages and opportunities for further research. Here, we will scratch the surface of what these models can do.

## EDA

We will be using the MIT Beth Israel Hospital (BIH) electrocardiogram dataset for the model comparison. This dataset contains about 110,000 labeled data points. Each sample is annotated as either normal (0), supraventricular premature beat (1), premature ventricular contraction (2), fusion of ventricular and normal beat (3), and finally unclassifiable beat (4). The ECGs were recorded at a frequency of 360 Hz. Thus, each sample was taken over 0.52 seconds since there are 187 measurements per sample. Data was downloaded from kaggle and can be found here: https://www.kaggle.com/shayanfazeli/heartbeat. A description of the data and the sampling procedure used can be found at the data's source: https://www.physionet.org/content/mitdb/1.0.0/.

The data is already cleaned and preprocessed for us making our lives much easier.

We can see that most of the samples are normal (0). The arrythmias are much less represented in the dataset. Hopefully, our model will still be able to make these classifications and do better than the baseline accuracy of 83%.

Some summaries of the characteristics of the different classes follow. Disclaimer: I am not a cardiologist. These observations are purely qualitative.

Normal heartbeats - sharp dropoff at beginning followed by small peak. Then, narrow spike before flatline.

Supraventricular premature beat, otherwise known as class 1 - more activity before main peak. After the main spike there is another bump.

Premature ventricular contraction or class 2 - this does not look healthy. A lot of stuff going on towards the front with short main peak.

Fusion of ventricular and normal beat: class 3 - everything compressed towards the front of the beat.

Finally, the unclassifiable beats: class 4 - each sample seems to deviate from the next significantly, which makes sense since these did not fit in any other category. A lot of jagged peaks.

## Network Building

We build both the ResNet and ODENet (the Neural ODE model). Source code can be found in the file models.py. PyTorch and torchdiffeq are heavily used. The latter library was written by the authors of Neural ODEs and includes diff eq solvers and functions for backpropagation. Our models are adapted from the mnist example for 1D classification. 

Both networks are made to be as similar as possible, so a fair comparison can be made. The downsampling layers and output layers are identical. Downsampling involves convolutions with a stride of two. The output layers use global average pooling followed by a linear layer. The only difference lies in the feature layers. For the ResNet, the feature layers consist of six residual blocks stacked end to end. Each residual block contains two of each of the following: convolution, group normalization, and ReLU activation. The ODENet's feature layers contain an almost identical layout to the residual blocks except that calls to a diff eq solver are made and backpropagation is achieved with the adjoint method. We use dopri5 as the default diff eq solver.

## Modeling

We train both models on the ECG training dataset for five epochs with a batch size of 128. We use PyTorch's F.cross_entropy as a loss function, which combines log softmax and negative log likelihood. We use stochastic gradient descent with momentum for the optimizer. 

Both models perform well on the test set with accuracies above 97%. This is significantly above the baseline accuracy of 83%. Therefore, the models both generalize well. The ResNet trained for only an hour while the ODENet trained for over seven hours. However, a benefit of the ODENet can be seen below. It has almost exactly 1/3 of the parameters as the ResNet and yet performed slightly better. This leads to the fact that Neural ODEs use constant memory (although with a high memory overhead due to the adjoint method).

## Conclusion

We compare the accuracy, speed, and memory of two state-of-the-art neural network models. The first, the ResNet, was introduced in 2015 and uses shortcut connections to improve performance. The second, the Neural ODE, was introduced in 2018 and built off of the ResNet by taking that model to its continuous limit. It utilizes differential equation solvers to solve for the output of the model given the input and the adjoint method to backpropagate. 

Overall, the ResNet and Neural ODE architectures perform nearly identically on our testing dataset. The main tradeoff is between speed and memory. The ResNet is significantly faster while the ODENet uses significantly fewer tunable parameters. The tradeoff is expected given the findings in the original Neural ODE paper.

In the future, we would like to investigate the more advanced uses of Neural ODEs, such as continuous normalizing flows, which is where they really seem to thrive. Also, we will look at some of the more recent research with Neural ODEs and how they are applied including but not limited to Augmented Neural ODEs and Stochastic Neural ODEs.

Much thanks to the authors of https://arxiv.org/abs/1512.03385 and https://arxiv.org/abs/1806.07366 as well as https://www.depthfirstlearning.com/2019/NeuralODEs for providing a curriculum to learn about the latest in neural networks.
