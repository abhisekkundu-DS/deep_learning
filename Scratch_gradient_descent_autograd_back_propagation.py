import numpy as np

# f = w * x
#f = 2 * x

X = np.array([1,2,3,4])
Y = np.array([2,4,6,8])

w = 0.0
#model predict
def forward(x):
  return w*x

#loss = MSE
def  loss(y, y_predict):
  return ((y_predict - y)**2).mean()

# gradient
#MSE = 1/N * (w*x -y)
# dj/dw = 1/N 2x (w*x - y)

def gradient(x,y, y_predict):
  return np.dot(2*x, y_predict -y).mean()

print(f'prediction before traning : f(5) = {forward(5)}:.3f}')

#Traning 
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
  #prediction = forward pass
  y_pred = forward(X)
  #loss
  l = loss(Y, y_pred)
  # gradients
  dw = gradient(X,Y,y_pred)
  #updated_ weight
  w-= learning_rate * dw

if epoch % 1 == 0:
  print(f'epoch {epoch + 1 }: w = {w: 3f} , loss = {l:.8f}')

print(f'prediction before traning : f(5) = {forward(5)}:.3f}')
  
  
  


