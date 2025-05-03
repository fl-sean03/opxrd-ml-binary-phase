import numpy as np

def interpolate_pattern(two_theta_values, intensities, new_two_theta_grid):
    """
    Interpolates pXRD intensities onto a new, fixed 2-theta grid.
    This is necessary to ensure all input patterns have the same length (number of features)
    for the machine learning model.

    Args:
        two_theta_values (list): List of original 2-theta values.
        intensities (list): List of original intensity values.
        new_two_theta_grid (np.ndarray): The target 2-theta grid for interpolation.

    Returns:
        np.ndarray: Interpolated intensity values on the new grid.
    """
    # Ensure inputs are numpy arrays for interpolation
    two_theta_values = np.asarray(two_theta_values)
    intensities = np.asarray(intensities)

    # Use numpy.interp for linear interpolation
    # xp: The x-coordinates of the data points, must be increasing.
    # fp: The y-coordinates of the data points, same length as xp.
    # x: The x-coordinates at which to evaluate the interpolated values.
    interpolated_intensities = np.interp(new_two_theta_grid, two_theta_values, intensities)

    return interpolated_intensities

def normalize_pattern(intensities):
    """
    Normalizes pXRD intensities by dividing by the maximum intensity value.
    This scales the intensities to a range between 0 and 1, which can help
    improve the performance and stability of some machine learning models.

    Args:
        intensities (np.ndarray): Array of intensity values.

    Returns:
        np.ndarray: Normalized intensity values.
    """
    max_intensity = np.max(intensities)
    if max_intensity > 0:
        normalized_intensities = intensities / max_intensity
    else:
        normalized_intensities = intensities # Avoid division by zero

    return normalized_intensities

if __name__ == "__main__":
    # Example Usage: Demonstrates the interpolation and normalization functions.
    print("Starting data preprocessing example...")

    # Create a sample pattern
    sample_two_theta = [10.0, 10.1, 10.2, 10.5, 11.0, 11.5, 12.0]
    sample_intensities = [5.0, 10.0, 8.0, 15.0, 12.0, 7.0, 3.0]

    print(f"Original 2-theta: {sample_two_theta}")
    print(f"Original Intensities: {sample_intensities}")

    # Define a new 2-theta grid (e.g., 10.0 to 12.0 with 0.1 step)
    new_grid = np.arange(10.0, 12.1, 0.1)
    print(f"\nNew 2-theta grid: {new_grid}")

    # Interpolate the pattern
    interpolated_intensities = interpolate_pattern(sample_two_theta, sample_intensities, new_grid)
    print(f"Interpolated Intensities: {interpolated_intensities}")

    # Normalize the interpolated pattern
    normalized_intensities = normalize_pattern(interpolated_intensities)
    print(f"Normalized Intensities: {normalized_intensities}")

    print("\nData preprocessing example finished.")