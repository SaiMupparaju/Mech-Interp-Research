import sys
import torch
import torch.nn as nn
sys.path.append('../')
from multimeter import CombinedNetwork

class HFCombinedNetwork(CombinedNetwork):
    '''
    HuggingFace based multimeter network. Overleads the original multimeter network.
    '''  
    def forward(self, input_ids, attention_mask = None, labels = None, **kwargs):
        #Attach representation hook to get output of start module
        self.representation_hook.attach(self.original_network)
        #Run original network all the way through to get the output.
        with torch.no_grad():
            original_output = self.original_network(input_ids)
        #Get frozen output from multimeter network
        frozen_output = self.frozen_network(self.representation_hook.representation)
        #Set modification hook frozen output
        self.modification_hook.frozen_output = frozen_output
        #Attach modification hook to the network
        self.modification_hook.attach(self.original_network)
        #Run through original network again with hook to recombine the entire frozen output
        final_output = self.original_network(input_ids)
        #Remove hooks from the network.
        self.representation_hook.remove()
        self.modification_hook.remove()
        return final_output