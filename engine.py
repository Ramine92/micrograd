import math
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
    
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data,_children = (self,other),op="+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward 

        return out

    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, _children = (self,other), op="*")

        def _backward():
            self.grad += other.grad * out.grad 
            other.grad += self.grad * out.grad 

        out._backward = _backward 

        return out

    def __rmul__(self,other):
        return self * other

    def tanh(self):
        x = self.data 
        t = (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
        out = Value(t,(self,),op="tanh")
        def _backward():
            self.grad += (1 - t**2) * out.grad
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
        

