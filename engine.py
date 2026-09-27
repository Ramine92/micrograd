import math
import numpy as np
class Value:
    def __init__(self,data,_children=(),op='',label='',grad=0.0):
        self.data = data
        self._prev = set(_children)
        self.grad = grad 
        self._backward = lambda : None
        self.label = label
        self._op = op 


    def __repr__(self):
        return f"Value | (data = {self.data})"

    def __neg__(self):
        return self * -1
    
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data,_children = (self,other),op="+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward 

        return out

    def __radd__(self,other):
        return self + other

    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, _children = (self,other), op="*")

        def _backward():
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad 

        out._backward = _backward 

        return out

    def __pow__(self,other):
        assert isinstance(other,(int,float)) , "only supporting int/float power for now"
        out = Value(self.data ** other,_children=(self,),op=f"** {other}")

        def _backward():
            self.grad += (other * (self.data ** (other -1)) ) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x),_children=(self,),op="exp")

        def _backward():
            self.grad += out.data * out.grad 
        out._backward = _backward

        return out

    def __rmul__(self,other):
        return self * other
    
    def __sub__(self,other):
        return self + (-other)

    def __truediv__(self,other):
        return self * other**-1

    def __rsub__(self,other):
        return other + (-self)
    
    def tanh(self):
        x = self.data 
        t = (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
        out = Value(t,(self,),op="tanh")
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
    
    def log(self):
        x = self.data
        assert x > 0 , "must be positive"
        out = Value(math.log(x),_children=(self,),op="log")
        def _backward():
            self.grad += (1/x)*out.grad
        out._backward = _backward

        return out 

    def relu(self):
        out = Value(max(0,self.data),_children=(self,),op="ReLU") 
        def _backward():
            self.grad = (out.data > 0) * out.grad
        out._backward = _backward 
        
        return out 

    def sigmoid(self):
        eps = 1e-8
        x = self.data
        out = Value(np.clip(1/(1+math.exp(-x)),eps,1-eps),_children=(self,),op="sigmoid")
        def _backward():
            self.grad += out.data*(1-out.data)* out.grad 
        out._backward = _backward

        return out 

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
        

