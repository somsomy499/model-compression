class Pruner:
    def __init__(self, sparsity=0.5):
        self.sparsity = sparsity
    def prune(self, model, **kwargs):
        return model
