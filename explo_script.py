import numpy as np

class VarianceAdaptor:
    def __init__(self, target_variance=1.0):
        self.target_variance = target_variance
        self.original_mean = None
        self.original_variance = None

    def fit(self, data):
        # Calculate the mean and variance of the original data
        self.original_mean = np.mean(data)
        self.original_variance = np.var(data)

    def transform(self, data):
        if self.original_mean is None or self.original_variance is None:
            raise Exception("Fit the adaptor first by calling the fit method.")

        # Normalize the original data to zero mean and unit variance
        zero_mean_data = data - self.original_mean
        unit_variance_data = zero_mean_data / np.sqrt(self.original_variance)

        # Scale to the target variance and add the original mean back
        scaled_data = np.sqrt(self.target_variance) * unit_variance_data
        adjusted_data = scaled_data + self.original_mean

        return adjusted_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

# Example usage
data = [1.0, 2.0, 3.0, 4.0, 5.0]
adaptor = VarianceAdaptor(target_variance=4.0)

# Fit the adaptor to the original data
adaptor.fit(data)

# Transform the original data
adjusted_data = adaptor.transform(data)
print("Adjusted data:", adjusted_data)

# Fit and transform in one step
adjusted_data = adaptor.fit_transform(data)
print("Adjusted data:", adjusted_data)
