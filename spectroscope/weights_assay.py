from typing import Iterable, Type
import numpy as np
import pandas as pd
import torch.nn as nn
import torch as t

NUM_STEPS = 792
IVL = 24

class WeightsAssay:
    def measure(self, cls: Type[MNISTConvNet], steps: Iterable[int]):
        model = cls() 
        measurements = []

        for i in steps:
            # Load model from snapshot
            model.load_state_dict(t.load(f"./snapshots/model-{i}.pt"))

            # Measure metrics
            measurements.append(self._measure(model, i))

        return pd.DataFrame(measurements)
    
    def _measure(self, model, step):
        weight_norms = self.weight_norms(model)
        singular_values = self.singular_values(model)
        max_singular_values = [max(sv_layer) for sv_layer in singular_values]

        return {
            "step": step,
            "weight_norms": weight_norms,
            "total_weight_norm": sum([wn ** 2 for wn in weight_norms]) ** 0.5,
            "singular_values": singular_values,
            "max_singular_values": max_singular_values, #maximum singular value in each layer
            "prod_max_singular_values": np.prod(max_singular_values),
            # "last_layer_rank": self.last_layer_rank(model),
        }
    
    @classmethod
    def weight_norms(cls, model: nn.Module):
        norms = []

        for p in model.parameters():
            norms.append(p.norm().item())

        return norms

    # This method finds the singular values for each layer in the network
    @classmethod
    def singular_values(cls, model: nn.Module):
        singular_values = []
        for name, param in model.named_parameters():
            if "weight" in name:
                if "conv" in name:
                    reshaped_param = param.data.view(param.size(0), -1)
                elif "fc" in name:
                    reshaped_param = param.data
                _, s, _ = t.svd(reshaped_param)
                singular_values.append(s.tolist())
        
        return singular_values
        
    # @classmethod
    # def last_layer_rank(cls, model: nn.Module):
    #    return t.linalg.matrix_rank(model.fc1.weight).item()
    # NOTE Nevermind this is full rank

    # TODO: Add more probes