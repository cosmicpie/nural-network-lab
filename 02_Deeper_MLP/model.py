import numpy as np

class NN2:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.w1 = np.array([])
        self.b1 = np.array([])
        self.w2 = np.array([])
        self.b2 = np.array([])
        self.w3 = np.array([])
        self.b3 = np.array([])

    def sigmoid(self,a):
        return 1/(1 + np.exp(-a))

    def sigmoid_derivative(self,a):
        return a * (1 - a)

    def relu(self,a):
        return np.maximum(0,a)

    def relu_derivative(self,a):
        return np.where(a > 0,1,0)

    def train(self):
        np.random.seed(42)

        self.w1 = np.random.randn(3,5)
        self.b1 = np.zeros((1,5))
        self.w2 = np.random.randn(5,4)
        self.b2 = np.zeros((1,4))
        self.w3 = np.random.randn(4,1)
        self.b3 = np.zeros((1,1))

        learning_rate = 0.1

        for i in range(100000):

            z1 = np.dot(self.x,self.w1) + self.b1
            a1 = self.relu(z1)

            z2 = np.dot(a1,self.w2) + self.b2
            a2 = self.relu(z2)

            z3 = np.dot(a2,self.w3) + self.b3
            a3 = self.sigmoid(z3)

            if i % 10000 == 0:
                loss = np.mean(np.square(self.y - a3))  # Mean Squared Error formula
                print(f"Iteration {i:6d} ---> Loss: {loss:.6f}")

            error1 = self.y - a3
            dz3 = error1 * self.sigmoid_derivative(a3)

            error2 = np.dot(dz3,self.w3.T)
            dz2 = error2 * self.relu_derivative(z2)

            error3 = np.dot(dz2,self.w2.T)
            dz1 = error3 * self.relu_derivative(z1)

            self.w1 += learning_rate * np.dot(self.x.T,dz1)
            self.b1 += np.sum(dz1, axis=0 , keepdims=True ) * learning_rate

            self.w2 += learning_rate * np.dot(a1.T,dz2)
            self.b2 += learning_rate * np.sum(dz2, axis=0 , keepdims=True)

            self.w3 += learning_rate * np.dot(a2.T,dz3)
            self.b3 += learning_rate * np.sum(dz3, axis=0 , keepdims=True)

        return print("""
!!!!!!!!!!!!! training complete.... !!!!!!!!!!!!!
use model.prediction(x) to predict 
        """)

    def wb(self):
              return print(f"""
                w1 is :   {self.w1}
                b1 is :   {self.b1}
                w2 is :   {self.w2}
                b2 is :   {self.b2}
                w3 is :   {self.w3}
                b3 is :   {self.b3}
              """)

    def prediction(self,x):
            z1 = np.dot(x,self.w1) + self.b1
            a1 = self.relu(z1)
            
            z2 = np.dot(a1,self.w2) + self.b2
            a2 = self.relu(z2)
            
            z3 = np.dot(a2,self.w3) + self.b3
            a3 = self.sigmoid(z3)

            pred = np.round(a3)

            print("-" * 50)
            print(f"raw prediction is :-  {a3[0][0]:.8f}")
            print(f"rounded pred is   :-  {int(pred[0][0])}")
            print("-" * 50)
        
