# Introduction 
In this lab we will go over the example provided in the [Deepwave](https://github.com/ar4/deepwave/tree/master/deepwave) page in [thie notebook](https://colab.research.google.com/drive/1PMO1rFAaibRjwjhBuyH3dLQ1sW5wfec_) with few modiifications. We won't apply a source inversion as in the original notebook and we won't normalize the data, but the gradient.

We assume you have some knowledge in coding with pytorch as deepwave is built using pytorch framework.  


# Deepwave FWI and pytorch training 
A simple code for training a model using pytorch consists of defining: 
 1. model
 2. optimizer 
 3. criterion 
 4. training loop 
 
A normal training example is below:

```python
import torch

# Define the model 
model = Netowrk(some_parameters>)

# Define the criterion 
criterion = torch.nn.MSELoss()

#Define the optimizer
optimizer = torch.optim.Adam(lr=0.0001)

# Training loop 
for epoch in range(num_epoch):
  for batch in range(num_batch):
     y_pred = model(x) # Forward pass
     loss   =  criterion(y_pred,y_true)
     loss.backward  # compute gradient 
     optimizer.step() # update the model 
```

The same workflow applies in Deepwave_FWI except we need to define the forward pass (In normal pytorch training, this is defined inside the network class). The forward pass in our case is a wavefield propegator. Thus the above code become like: 

```python
# Inversion (training) loop 
for epoch in range(num_epoch):
  for batch in range(num_batch):
     prop = deepwave.scalar.Propagator({'vp': model}, dx)  # since the model is changing we redefine the propegator every batch
     y_pred = prop(wavelet, x_s, x_r, dt)  # Forward pass takes wavelet, sources/receiver coordinates, sampling rate
     loss   =  criterion(y_pred,y_true)
     loss.backward  # compute gradient 
     optimizer.step() # update the model 
```

