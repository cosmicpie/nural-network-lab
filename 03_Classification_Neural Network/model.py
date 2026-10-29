import numpy as np

class NN3:
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
        return np.where(a > 0,1,0)

    def softmax(self,a):
        exp = np.exp(a - np.max(a, axis=1 , keepdims=True))
        output = exp / np.sum(exp, axis=1 , keepdims=True)
        return output
        
    
    def train(self):
        np.random.seed(42)

        self.w1 = np.random.randn(4,8)
        self.b1 = np.zeros((1,8))
        self.w2 = np.random.randn(8,4)
        self.b2 = np.zeros((1,4))
        self.w3 = np.random.randn(4,3)
        self.b3 = np.zeros((1,3))

        learning_rate = 0.1
        
        for i in range(70000):

            z1 = np.dot(self.x,self.w1) + self.b1
            a1 = self.relu(z1)

            z2 = np.dot(a1,self.w2) + self.b2
            a2 = self.relu(z2)

            z3 = np.dot(a2,self.w3) + self.b3
            a3 = self.softmax(z3)

            if i % 10000 == 0:
               loss = -np.mean(
               np.sum(self.y * np.log(a3 + 1e-8), axis=1)
                )

               predictions = np.argmax(a3, axis=1)
               actual = np.argmax(self.y, axis=1)

               accuracy = np.mean(predictions == actual)

               print(f"Iteration: {i} | Loss: {loss:.4f} | Accuracy: {accuracy:.2%}")

            dz3 = (a3 - self.y)/ self.x.shape[0]

            error1 = np.dot(dz3,self.w3.T)
            dz2 = error1 * self.relu_derivative(z2)

            error2 = np.dot(dz2,self.w2.T)
            dz1 = error2 * self.relu_derivative(z1)

            self.w1 -= learning_rate * np.dot(self.x.T,dz1)
            self.b1 -= learning_rate * np.sum(dz1, axis=0, keepdims=True)

            self.w2 -= learning_rate * np.dot(a1.T,dz2)
            self.b2 -= learning_rate * np.sum(dz2, axis=0, keepdims=True)

            self.w3 -= learning_rate * np.dot(a2.T,dz3)
            self.b3 -= learning_rate * np.sum(dz3, axis=0, keepdims=True)

    def prediction(self,x):
        z1 = np.dot(x,self.w1) + self.b1
        a1 = self.relu(z1)     
        z2 = np.dot(a1,self.w2) + self.b2
        a2 = self.relu(z2)
        z3 = np.dot(a2,self.w3) + self.b3
        a3 = self.softmax(z3)

        return np.argmax(a3, axis=1)
    
    