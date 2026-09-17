class Value:
    def __init__(self,data,_children=(),op='',label='',grad=0.0):
        self.data = data
        self._prev = set(_children)
        self.grad = grad 
        self.label = label
        self._op = op 


    def __repr__(self):
        return f"Value | (data = {self.data})"
    
    def __add__(self,other):
        return Value(self.data + other.data,_children = (self,other),op="+")

    def __mul__(self,other):
        return Value(self.data * other.data, _children = (self,other), op="*")

