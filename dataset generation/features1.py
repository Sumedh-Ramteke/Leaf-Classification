import cv2
import numpy as np

# Function to compute brightness (mean)
def brightness(image):
    return np.mean(image)

# Function to compute contrast (standard deviation)
def contrast(image):
    return np.std(image)

# Function to compute skewness
def skewness(image):
    mean = np.mean(image)
    return np.sum((image - mean) ** 3) / (image.size * np.std(image) ** 3)

# Function to compute kurtosis
def kurtosis(image):
    mean = np.mean(image)
    return np.sum((image - mean) ** 4) / (image.size * np.std(image) ** 4) - 3

# Function to compute entropy (Shannon)
def entropy(image):
    histogram, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    histogram = histogram / np.sum(histogram)  # Normalize histogram
    return -np.sum(histogram * np.log2(histogram + 1e-10))  # Avoid log(0)

# Function to compute energy
def energy(image):
    histogram, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    histogram = histogram / np.sum(histogram)  # Normalize histogram
    return np.sum(histogram ** 2)

# Function to compute absolute moments
def absolute_moment(image, k):
    return np.sum(np.abs(image) ** k) / image.size

# GLCM calculation
def calculate_glcm(image, d_row, d_col):
    if len(image.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")

    # Clipping image pixel values to [0, 255] range
    image = np.clip(image, 0, 255).astype(np.uint8)

    # Initialize GLCM matrix for 256 gray levels
    glcm = np.zeros((256, 256), dtype=np.float64)

    rows, cols = image.shape
    for i in range(rows - d_row):
        for j in range(cols - d_col):
            row_val = image[i][j]
            col_val = image[i + d_row][j + d_col]
            if row_val <= 255 and col_val <= 255:
                glcm[row_val][col_val] += 1

    # Normalize the GLCM
    glcm_sum = np.sum(glcm)
    if glcm_sum > 0:
        glcm /= glcm_sum
    else:
        raise ValueError("GLCM calculation resulted in a zero matrix.")

    return glcm

# Texture Features from GLCM
def angular_second_moment(glcm):
    return np.sum(glcm ** 2)

def contrast_from_glcm(glcm):
    return np.sum([(i - j) ** 2 * glcm[i, j] for i in range(glcm.shape[0]) for j in range(glcm.shape[1])])

def inverse_difference_moment(glcm):
    return np.sum(glcm / (1 + (np.arange(glcm.shape[0])[:, None] - np.arange(glcm.shape[1])) ** 2))

def entropy_from_glcm(glcm):
    return -np.sum(glcm * np.log(glcm + 1e-10))

def correlation_from_glcm(glcm):
    mean_x = np.sum(glcm * np.arange(glcm.shape[0]).reshape(-1, 1))
    mean_y = np.sum(glcm * np.arange(glcm.shape[1]).reshape(1, -1))
    std_x = np.sqrt(np.sum((np.arange(glcm.shape[0]) - mean_x) ** 2 * np.sum(glcm, axis=1)))
    std_y = np.sqrt(np.sum((np.arange(glcm.shape[1]) - mean_y) ** 2 * np.sum(glcm, axis=0)))
    return np.sum((glcm - mean_x) * (glcm.T - mean_y)) / (std_x * std_y)

def variance_from_glcm(glcm):
    return np.var(glcm)

def sum_average(glcm):
    row_sums = np.sum(glcm, axis=1)
    col_sums = np.sum(glcm, axis=0)
    return np.sum(np.arange(glcm.shape[0]) * (row_sums + col_sums))

def sum_variance(glcm, sum_average_val):
    row_sums = np.sum(glcm, axis=1)
    col_sums = np.sum(glcm, axis=0)
    return np.sum((np.arange(glcm.shape[0]) - sum_average_val) ** 2 * (row_sums + col_sums))

def sum_entropy(glcm):
    return -np.sum(np.sum(glcm, axis=1) * np.log(np.sum(glcm, axis=1) + 1e-10))

def difference_average(glcm):
    return np.mean(np.abs(np.arange(glcm.shape[0])[:, None] - np.arange(glcm.shape[1])) * glcm)

def difference_variance(glcm):
    return np.var(np.sum(glcm, axis=1))

def difference_entropy(glcm):
    return -np.sum(np.sum(glcm, axis=1) * np.log(np.sum(glcm, axis=1) + 1e-10))

def information_measure_I(glcm):
    hx = entropy_from_glcm(glcm)
    hy = sum_entropy(glcm)
    return hx / max(hx, hy)

def information_measure_II(glcm):
    return np.log(np.sum(np.sum(glcm)))

def maximal_correlation_coefficient(glcm):
    return np.max(glcm)

def short_run_emphasis(glcm):
    return np.sum(glcm[glcm < 2])  # For example if 2 is cosidered

def long_run_emphasis(glcm):
    return np.sum(glcm[glcm > 2])  # Same criterion as above

def gray_level_nonuniformity(glcm):
    return np.sum(np.var(glcm, axis=1))

# Co-occurrence Matrix (Normalized)
def co_occurrence_matrix(image):
    glcm = calculate_glcm(image, 1, 1)
    return glcm / np.sum(glcm)

# Covariance Matrix
def covariance_matrix(image):
    mean = np.mean(image)
    return np.cov(image.flatten())  # Covariance of the flattened image

# Difference of Entropy from glcm
def difference_entropy(glcm):
    diff_entropy = 0
    max_gray_level = glcm.shape[0]

    for d in range(max_gray_level):  # Iterate over the differences
        for i in range(max_gray_level):
            j = i - d  # Compute the corresponding j
            if 0 <= j < max_gray_level:  # Ensure j is within bounds
                p_ij = glcm[i, j]
                if p_ij > 0:  # Only consider positive probabilities
                    diff_entropy -= p_ij * np.log(p_ij + 1e-10)  # Add small value to avoid log(0)

    return diff_entropy

# Eigen Value(Second largest Eigenvalue)
def second_largest_eigenvalue(glcm):
    # Ensure the GLCM is normalized
    glcm_normalized = glcm / np.sum(glcm) if np.sum(glcm) > 0 else glcm
    
    # Calculate eigenvalues
    eigenvalues, _ = np.linalg.eig(glcm_normalized)

    return np.partition(eigenvalues.real, -2)[-2]  # Get the second largest eigenvalue


def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: Unable to load image {image_path}.")
        return None

    # Check the shape of the image
    if image.size == 0:
        print(f"Warning: Image at {image_path} is empty.")
        return None

    # Resize image for consistent feature extraction
    image = cv2.resize(image, (250, 250))
   
    features = {}
    
    features['Brightness'] = brightness(image)
    features['Contrast'] = contrast(image)
    features['Mean'] = np.mean(image)
    features['Variance'] = np.var(image)
    features['Skewness'] = skewness(image)
    features['Kurtosis'] = kurtosis(image)
    features['Entropy'] = entropy(image)
    features['Energy'] = energy(image)
    features['Absolute Moment k=1'] = absolute_moment(image, 1)
    features['Absolute Moment k=2'] = absolute_moment(image, 2)
    
    glcm = calculate_glcm(image, 1, 1)
    features['ASM'] = angular_second_moment(glcm)
    features['Contrast (GLCM)'] = contrast_from_glcm(glcm)
    features['IDF'] = inverse_difference_moment(glcm)
    features['Entropy (GLCM)'] = entropy_from_glcm(glcm)
    features['Correlation (GLCM)'] = correlation_from_glcm(glcm)
    features['Variance (GLCM)'] = variance_from_glcm(glcm)
    features['Sum Average'] = sum_average(glcm)
    features['Sum Variance'] = sum_variance(glcm, features['Sum Average'])
    features['Sum Entropy'] = sum_entropy(glcm)
    features['Difference Average'] = difference_average(glcm)
    features['Difference Variance'] = difference_variance(glcm)
    features['Difference Entropy'] = difference_entropy(glcm)
    features['Information Measure I'] = information_measure_I(glcm)
    features['Information Measure II'] = information_measure_II(glcm)
    features['Maximal Correlation Coefficient'] = maximal_correlation_coefficient(glcm)
    features['Short-run Emphasis'] = short_run_emphasis(glcm)
    features['Long-run Emphasis'] = long_run_emphasis(glcm)
    features['Gray-level Nonuniformity'] = gray_level_nonuniformity(glcm)
    features['Difference of Entropy'] = difference_entropy(glcm)
    features['Second Largest Eigenvalue'] = second_largest_eigenvalue(glcm)

    return features