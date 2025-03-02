import torch
import torch.nn as nn
from hooks import RepresentationHook, ModificationHook

class CombinedNetwork(nn.Module):
    '''
    Multimeter network. Takes an original network and trains with input from a frozen network. 
    Inputs:
        original_network: Optimized network
        frozen_network: Multimeter network that original network can steal weights from
        representation_layer_index: start layer index to connect multimeter to.
        modification_layer_index: end layer index to combine frozen output and original network output.
    '''
    def __init__(self, original_network, frozen_network, representation_layer_index, modification_layer_index):
        super().__init__()
        self.original_network = original_network
        self.frozen_network = frozen_network
        self.parameter = nn.Parameter(torch.randn(()))
        self.representation_hook = RepresentationHook(representation_layer_index)
        self.modification_hook = ModificationHook(modification_layer_index, self.parameter)
    
    def forward(self, x):
        #Attach representation hook to get output of start module
        self.representation_hook.attach(self.original_network)
        #Run original network all the way through to get the output.
        with torch.no_grad():
            original_output = self.original_network(x)
        #Get frozen output from multimeter network
        frozen_output = self.frozen_network(self.representation_hook.representation)
        #Set modification hook frozen output
        self.modification_hook.frozen_output = frozen_output
        #Attach modification hook to the network
        self.modification_hook.attach(self.original_network)
        #Run through original network again with hook to recombine the entire frozen output
        final_output = self.original_network(x)
        #Remove hooks from the network.
        self.representation_hook.remove()
        self.modification_hook.remove()
        return final_output