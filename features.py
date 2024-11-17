import cv2
import numpy as np
from scipy.fftpack import fft
from scipy.stats import skew
import mahotas

# Function to compute brightness (mean)
def brightness(image):
    return np.mean(image)

# Function to compute contrast (standard deviation)
def contrast(image):
    return np.std(image)

# Function to compute skewness
def skewness(image):
    return skew(image.flatten())

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

def zernike_moments(image, radius=250, degree=8):
    h, w = image.shape
    crop_size = min(h, w)
    image_cropped = image[:crop_size, :crop_size]

    image_cropped = image_cropped / 255.0

    moments = mahotas.features.zernike_moments(image_cropped, radius=radius, degree=degree)
    return moments

# Fast Fourier Transform (FFT)
def fft_features(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    return np.mean(magnitude_spectrum), np.std(magnitude_spectrum)

# Chaincode calculation
def chaincode(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour = contours[0].reshape(-1, 2)
        chain = []
        for i in range(1, len(contour)):
            dx = contour[i][0] - contour[i-1][0]
            dy = contour[i][1] - contour[i-1][1]
            angle = np.arctan2(dy, dx) * 180 / np.pi
            chain.append(angle)
        return np.mean(chain)
    return None

# Function to compute eccentricity
def eccentricity(image):
    moments = cv2.moments(image)
    if moments["mu02"] != 0:
        value = 1 - (moments["mu20"] / moments["mu02"])
        if value < 0:
            # Log a warning or handle gracefully
            print(f"Warning: Negative value encountered for eccentricity calculation: {value}")
            return 0  # Return a default value for negative inputs
        ecc = np.sqrt(value)
        return ecc
    return 0

# Function to compute orientation
def orientation(image):
    moments = cv2.moments(image)
    return 0.5 * np.arctan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"])

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

    features['Zernike Moment'] = zernike_moments(image, 2, 2)  # Example, n=2, m=2
    features['FFT Mean'] = fft_features(image)[0]
    features['FFT Std'] = fft_features(image)[1]
    features['Chaincode'] = chaincode(image)
    features['Eccentricity'] = eccentricity(image)
    features['Orientation'] = orientation(image)

    return features