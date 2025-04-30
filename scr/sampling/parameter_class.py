from dataclasses import dataclass
import numpy as np


@dataclass
class Parameter:
    name: str
    param_range: list
    unit : str = None
    is_log: bool = False

    def scale_from_lhs(self, lhs_samples: np.ndarray) -> np.ndarray:
        """Scale values from lhs (between 0 and 1) to parameter ranges.

        Args:
            lhs_samples (np.ndarray): 

        Returns:
            _type_: scaled samples
        """        
        assert self.param_range[0] < self.param_range[1], (
            f"{self.name}: Range should be in ascending order"
        )

        if self.is_log:
            param_range = np.log10(self.param_range)
            low, high = param_range[0], param_range[1]
        else:
            low, high = self.param_range[0], self.param_range[1]

        samples = (high - low) * lhs_samples + low

        if self.is_log:
            samples = 10**samples

        return samples

    def mean(self):
        if self.is_log:
            return np.mean(np.log10(self.param_range))
        else:
            return np.mean(self.param_range)