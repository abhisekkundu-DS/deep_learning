# 1) Design moedel (input, output, forward pass with different layer)
# 2) Construct loss and optimizer
# 3) Traning loop
#    - Forward = compute prediction and loss
#    - Backward = compute gradients 
#    - Update weights

import torch
import torch.nm as nn

# Linear regression 
# f = w * x

# here : f = 2 * x

# 0) Traning samples
X = troch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

# 1) Design Model : weights to optimize and forward function
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
  return w * x

print(f'Predictions before training : f(5) = {forward(5).item():.3f}')

# 2) Define loss and optimizer
learning_rate = 0.01
n_iters = 100

# callable function
loss = nn.MSELoss()


optimizer = torch.optim.SGD([w], lr=learning_rate)

# 3) Traning loop
for epoch in range(n_iters):
  #predict = forward pass
  y_predicted = forward(X)

  #loss
  l = loss(Y, y_predicted)
  #calculate gradients = backward passs
  l.backward()

  #update weights
  optimizer.step()

  #Zero the gradients after updating
  optimizer.zero_grad()

  if epoch % 10 == 0:
    print('epoch ', epoch+1, ': w = ', w, ' loss = ', l)

print(f'Prediction after training: f(5) = {forward(5).item():.3f}')
    
