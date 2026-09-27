from engine import Value 
import random

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []
class Neuron(Module):
    def __init__(self,nin: float,nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
        self.nonlin = nonlin
    def __call__(self,x):
        act = sum((wi * xi for wi,xi in zip(self.w,x)),0.0)+self.b 
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'tanh' if self.nonlin else 'Linear'} Neuron({len(self.w)})"

class Layer:
    def __init__(self,nin: float,nout : float,**kwargs):
        self.neurons = [Neuron(nin,**kwargs) for _ in range(nout)]

    def __call__(self,x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out 

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
class MLP(Module):
    def __init__(self,nin: float , nouts : list):
       sz = [nin] + nouts
       self.layers = [Layer(sz[i],sz[i+1],nonlin=i != len(nouts)-1) for i in range(len(nouts))]

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def __repr__(self):
        return f"MLP of : [{', '.join(str(layer) for layer in self.layers)}]"