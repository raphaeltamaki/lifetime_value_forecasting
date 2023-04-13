import numpy as np
from scipy.special import erfc
from scipy.stats import pareto, lognorm

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

    def __init__(self, reference_volume: float, reference_cpi: float, average: float, standard_deviation: float=0.5):
        self.reference_volume = reference_volume
        self.average = average
        self.standard_deviation = standard_deviation

        self.ref_prob = self.calculate_cdf(reference_cpi)
        
    def calculate_cdf(self, x: float):
        return lognorm.cdf(x, scale=np.exp(self.average), s=self.standard_deviation)

    def calculate_pdf(self, x: float):
        return lognorm.pdf(x, scale=np.exp(self.average), s=self.standard_deviation)
    
    def calculate_volume(self, cpi: float):
        
        return self.reference_volume * self.calculate_cdf(cpi) / self.ref_prob
        
    

class CumulativeParetoVolume(BaseVolume):

    def __init__(self, reference_volume: float, reference_cpi: float, scale: float):
        
        self.reference_volume = reference_volume
        self.scale = scale
        self.ref_prob = pareto.cdf(reference_cpi + 1, scale) # +1 to start from 0 the probability for pareto
    
    def calculate_cdf(self, x: float):
        return pareto.cdf(x + 1, self.scale)
    
    def calculate_pdf(self, x: float):
        return pareto.pdf(x + 1, self.scale)

    def calculate_volume(self, cpi: float):
        return self.reference_volume * pareto.cdf(cpi + 1, self.scale) / self.ref_prob # +1 to start from 0 the probability for pareto
