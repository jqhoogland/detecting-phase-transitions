from typing import Iterable


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

        return {
            "step": step,
            "weight_norms": weight_norms,
            "total_weight_norm": sum([wn ** 2 for wn in weight_norms]) ** 0.5,
            # "last_layer_rank": self.last_layer_rank(model),
        }
    
    @classmethod
    def weight_norms(cls, model: nn.Module):
        norms = []

        for p in model.parameters():
            norms.append(p.norm().item())

        return norms
        
    # @classmethod
    # def last_layer_rank(cls, model: nn.Module):
    #    return t.linalg.matrix_rank(model.fc1.weight).item()
    # NOTE Nevermind this is full rank

    # TODO: Add more probes