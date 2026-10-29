import numpy as np

X = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0]
], dtype=np.float32)

y = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
    [1.0],
    [0.0],
    [0.0],
    [1.0]
], dtype=np.float32)

class NN1:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.w1 = np.array([])
        self.b1 = np.array([])
        self.w2 = np.array([])
        self.b2 = np.array([])


    def train(self):
      
      np.random.seed(42)

      self.w1 = np.random.randn(3,3)
      self.b1 = np.zeros((1,3))

      self.w2 = np.random.randn(3,1)
      self.b2 = np.zeros((1,1))

      def sigmoid(x):
       return 1 / (1 + np.exp(-x))

      def sigmoid_derivative(x):
       return x * ( 1 - x )

      learning_rate = 0.1

      for i in range(100000):

        z1 = np.dot(self.x,self.w1) + self.b1
        a1 = sigmoid(z1)

        z2 = np.dot(a1,self.w2) + self.b2
        a2 = sigmoid(z2)

        error = self.y - a2
        dz2 = error * sigmoid_derivative(a2)

        error2 = np.dot(dz2,self.w2.T)
        dz1 = error2 * sigmoid_derivative(a1)

        self.w1 += np.dot(self.x.T,dz1) * learning_rate
        self.b1 += np.sum(dz1, axis=0 , keepdims=True) * learning_rate

        self.w2 += np.dot(a1.T,dz2) * learning_rate
        self.b2 += np.sum(dz2, axis=0 , keepdims=True) * learning_rate

    def wb(self):
      return print(f"""
        w1 is :   {self.w1}
        b1 is :   {self.b1}
        w2 is :   {self.w2}
        b2 is :   {self.b2}
      """)

    def prediction(self,X):
      def sigmoid(x):
             return 1 / (1 + np.exp(-x))
      
      def sigmoid_derivative(x):
             return x * ( 1 - x )

      z1 = np.dot(X,self.w1) + self.b1
      a1 = sigmoid(z1)
      
      z2 = np.dot(a1,self.w2) + self.b2
      a2 = sigmoid(z2)

      pred = np.round(a2)

      print(f"prediction is:   {pred}")
