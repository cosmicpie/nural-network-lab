import numpy as np

class NN5:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.w1 = np.array([])
        self.b1 = np.array([])

        self.w2 = np.array([])
        self.b2 = np.array([])

        self.w3 = np.array([])
        self.b3 = np.array([])


    def relu(self,a):
        return np.maximum(0,a)

    def relu_derivative(self,a):
        return np.where(a >0,1,0)



    

    def train(self):
        np.random.seed(42)
    
        learning_rate = 0.001

        l2_loss = 0.1

        self.w1 = np.random.randn(4,16) * np.sqrt(2/4)
        self.b1 = np.zeros((1,16))

        self.w2 = np.random.randn(16,8) * np.sqrt(2/16)
        self.b2 = np.zeros((1,8))

        self.w3 = np.random.randn(8,1) * np.sqrt(2/8)
        self.b3 = np.zeros((1,1))


        for i in range(100000):

          z1 = np.dot(self.x,self.w1) + self.b1
          a1 = self.relu(z1)

          z2 = np.dot(a1,self.w2) + self.b2
          a2 = self.relu(z2)

          z3 = np.dot(a2,self.w3) + self.b3
          a3 = z3
 
          mse = np.mean((a3 - self.y)**2)

          l2_panalty = (l2_loss/2) * (np.sum(self.w1**2) + np.sum(self.w2**2) + np.sum(self.w3**2))

          loss = mse + l2_panalty

          dz3 = 2*(a3 - self.y)/ self.x.shape[0]
 
          error1 = np.dot(dz3,self.w3.T)
          dz2 = error1 * self.relu_derivative(z2)

          error2 = np.dot(dz2,self.w2.T)
          dz1 = error2 * self.relu_derivative(z1)

          dw1 = np.dot(self.x.T,dz1) + l2_loss * self.w1
          dw2 = np.dot(a1.T,dz2) + l2_loss * self.w2
          dw3 = np.dot(a2.T,dz3) + l2_loss *self.w3

          self.w1 -= learning_rate * dw1
          self.w2 -= learning_rate * dw2
          self.w3 -= learning_rate * dw3

          self.b1 -= np.sum(dz1, axis=0 , keepdims=True) * learning_rate
          self.b2 -= np.sum(dz2, axis=0 , keepdims=True) * learning_rate
          self.b3 -= np.sum(dz3, axis=0, keepdims=True) * learning_rate

          if i % 10000 == 0:
             print(f"Iteration: {i} | Loss: {loss:.4f}")
             print("z1 min:", np.min(z1))
             print("z1 max:", np.max(z1))
             print("a1 mean:", np.mean(a1))
             print("z2 min:", np.min(z2))
             print("z2 max:", np.max(z2))
             print("a2 mean:", np.mean(a2))
             print(f"MSE: {mse:.6f} | L2: {l2_panalty:.6f} | Total: {loss:.6f}")
                
    def predict(self,x):
        z1 = np.dot(x,self.w1) + self.b1
        a1 = self.relu(z1)

        z2 = np.dot(a1,self.w2) + self.b2
        a2 = self.relu(z2)

        z3 = np.dot(a2,self.w3) + self.b3
        a3 = z3
        print(f"your prediction is    {a3}")