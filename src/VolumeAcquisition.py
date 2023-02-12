import numpy as np
from scipy.special import erfc

class BaseVolume():
    
    def __init__(self, reference_volume: float, reference_spend: float):
        self.reference_volume = reference_volume 
        self.reference_spend = reference_spend

    def calculate_volume(self, cpi: float):
        raise NotImplementedError
        

class LinearVolume(BaseVolume):

    def calculate_volume(self, cpi: float):
        cpi_reference =  self.reference_spend / self.reference_volume
        return self.reference_volume * cpi / cpi_reference


class QuadraticVolume(BaseVolume):

    def calculate_volume(self, cpi: float):
        cpi_reference =  self.reference_spend / self.reference_volume
        return self.reference_volume * cpi**2 / cpi_reference**2


class CumulativeLognormalVolume(BaseVolume):

    def __init__(self, average: float, standard_deviation: float):
        self.average = average
        self.standard_deviation = standard_deviation
        self.expected_value = np.exp(average + 0.5*standard_deviation**2)

    def calculate_volume(self, cpi: float):
        
        return 0.5 * erfc((self.average - np.log(cpi)) * 0.707107 / self.standard_deviation)
