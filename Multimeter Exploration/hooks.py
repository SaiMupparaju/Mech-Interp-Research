import torch

class RepresentationHook:
    '''
    Attaches hook to module associated with a layer_index in pytorch. The layer_index indexes the flattened modules in the pytorch
    model where the leaf modules are flattened. It then grabs the representations using the a hook function.
    Inputs:
        layer_index: Index to grab representations from. Set based on flattened leaf modules list designed by the network.
    '''
    def __init__(self, layer_index):
        self.layer_index = layer_index
        self.representation = None
        self.handle = None
        self.leaf_modules = []

    def hook_fn(self, module, input, output):
        #Set a class attribute called `representation` to the output of the module.
        self.representation = output

    def attach(self, model):
        def get_leaf_modules(module):
            #Iterate over modules of pytorch. Some modules like nn.Sequential, nn.ModuleList have children and append those recursively.
            if not list(module.children()):
                self.leaf_modules.append(module)
            for child in module.children():
                get_leaf_modules(child)

        get_leaf_modules(model)
        #Register forward hook for the module that corresponds to the layer_index.
        if self.layer_index < len(self.leaf_modules):
            self.handle = self.leaf_modules[self.layer_index].register_forward_hook(self.hook_fn)
        else:
            raise IndexError(f'Layer index {self.layer_index} is out of range. Max index is {len(self.leaf_modules) - 1}')

    def remove(self):
        if self.handle is not None:
            self.handle.remove()

class ModificationHook:
    '''
    Attaches hook to module associated with layer_index in pytorch. Applies a modification on this hook based on a passed in representation from 
    an external network in the hook function. Applies a weighted average of this with the current output of the module.
    Inputs:
        layer_index: Index to grab representations from. Set based on flattened leaf modules list designed by the network.
        param: nn.Parameter set by pytorch and tuned to to incorporate the frozen output from another submodel with the current model.
    '''
    def __init__(self, layer_index, param):
        self.layer_index = layer_index
        self.frozen_output = None
        self.handle = None
        self.leaf_modules = []
        self.param = param

    def hook_fn(self, module, input, output):
        #Weighted sum based on the sigmoid of the passed in parameter. 
        return (1-torch.sigmoid(self.param)) * output + torch.sigmoid(self.param) * self.frozen_output

    def attach(self, model):
        #Iterate over modules of pytorch. Some modules like nn.Sequential, nn.ModuleList have children so iterate and append those recursively.
        def get_leaf_modules(module):
            if not list(module.children()):  # if leaf node
                self.leaf_modules.append(module)
            for child in module.children():
                get_leaf_modules(child)

        get_leaf_modules(model)
        #Register forward hook for the module that corresponds to the layer_index.
        if self.layer_index < len(self.leaf_modules):
            self.handle = self.leaf_modules[self.layer_index].register_forward_hook(self.hook_fn)
        else:
            raise IndexError(f'Layer index {self.layer_index} is out of range. Max index is {len(self.leaf_modules) - 1}')

    def remove(self):
        if self.handle is not None:
            self.handle.remove()