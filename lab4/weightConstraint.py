
import torch

class weightConstraint(object):
    def __init__(self):
        pass
    
    def __call__(self, module):
        
        if hasattr(module,'weight'):
            print("Entered")
            w=module.weight.data
            w = torch.clamp(w, max = 0.5)
            module.weight.data=w