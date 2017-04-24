import numpy as np
import matplotlib.pyplot as plt

def get_color_gradient_combined(grad_combined, color_combined):
    combined = np.zeros_like(grad_combined)
    combined[(grad_combined==1) | (color_combined==1)] = 1
    return combined

def draw_color_gradient_combined(combined, grad_combined, color_combined):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    ax1.imshow(combined, cmap='gray')
    stack_image = np.dstack(( np.zeros_like(grad_combined), grad_combined, color_combined))
    ax2.imshow(stack_image, cmap='gray')
    return None