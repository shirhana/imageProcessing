import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


#####################################
#                                   #
#         HELPFUL FUNCTIONS         # 
#                                   #
#####################################

def make_non_black_white(image_path, output_path):
    """
    Converts all non-black pixels in the image to white and saves the result.

    :param image_path: Path to the input image.
    :param output_path: Path to save the output image.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Define a threshold to detect black pixels (RGB values close to [0, 0, 0])
    threshold = 10
    non_black_pixels = np.any(image > threshold, axis=-1)  # Check if any channel is above the threshold

    # Set non-black pixels to white
    image[non_black_pixels] = [255, 255, 255]

    # Save the result
    cv2.imwrite(output_path, image)

def get_image_size(image_path):
    """
    Gets the size (dimensions) of an image.

    :param image_path: Path to the image file.
    """
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Get the shape of the image (height, width, channels)
    height, width = image.shape[:2]
    return width, height

def resize_image(image_path, new_width, new_height):
    """
    Resizes an image to the specified width and height.

    :param image_path: Path to the image file.
    :param new_width: Desired width for the resized image.
    :param new_height: Desired height for the resized image.
    :return: The resized image.
    """
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image

def resize_img_according_another_img(image_path, comparing_image_path, resized_image_path):
    width, height = get_image_size(comparing_image_path)
    resized_image = resize_image(image_path, width, height)

    # If the image was resized, save and display it
    if resized_image is not None:
        cv2.imwrite(resized_image_path, resized_image)


##############################################################

def create_gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid for the given image."""
    gaussian_pyramid = [image]
    for i in range(levels):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)
    return gaussian_pyramid

def create_laplacian_pyramid(gaussian_pyramid):
    """Creates a Laplacian pyramid from a Gaussian pyramid."""
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        upsampled = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i], upsampled)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])  # Add the smallest level
    return laplacian_pyramid

def blend_pyramids(laplacian_a, laplacian_b, gaussian_mask):
    """Blends two Laplacian pyramids using a Gaussian pyramid mask."""
    blended_pyramid = []
    for la, lb, gm in zip(laplacian_a, laplacian_b, gaussian_mask):
        if len(gm.shape) == 2:  # If gm is single-channel, convert to 3-channel
            gm = np.repeat(gm[:, :, np.newaxis], 3, axis=2)
        blended = gm * la + (1 - gm) * lb
        blended_pyramid.append(blended)
    return blended_pyramid

def reconstruct_from_pyramid(laplacian_pyramid):
    """Reconstructs the image from a Laplacian pyramid."""
    image = laplacian_pyramid[-1]
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
        image = cv2.pyrUp(image, dstsize=size)
        image = cv2.add(image, laplacian_pyramid[i])
    return image

def plot_laplacian_pyramid(laplacian_pyramid):
    """Plot all levels of a Laplacian pyramid."""
    levels = len(laplacian_pyramid)
    plt.figure()
    
    for i in range(levels):
        plt.subplot(1, levels, i+1)
        plt.imshow(cv2.cvtColor(laplacian_pyramid[i], cv2.COLOR_BGR2RGB))  # Convert to RGB for proper display
        plt.title(f'Level {i}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def blend_images(image1, image2, mask, levels=5):
    """Blends two images using a mask and the Multiresolution Pyramid Spline algorithm."""
    # Ensure all inputs are the same size
    image1 = cv2.resize(image1, (mask.shape[1], mask.shape[0]))
    image2 = cv2.resize(image2, (mask.shape[1], mask.shape[0]))
    
    # Normalize the mask to range [0, 1]
    mask = mask / 255.0 if mask.max() > 1 else mask

    # Create Gaussian and Laplacian pyramids
    gaussian_mask = create_gaussian_pyramid(mask, levels)
    laplacian_a = create_laplacian_pyramid(create_gaussian_pyramid(image1, levels))
    laplacian_b = create_laplacian_pyramid(create_gaussian_pyramid(image2, levels))


    # plot_laplacian_pyramid(laplacian_a)
    # Blend pyramids and reconstruct the final image
    blended_pyramid = blend_pyramids(laplacian_a, laplacian_b, gaussian_mask)
    blended_image = reconstruct_from_pyramid(blended_pyramid)

    # blended_image = reconstruct_from_pyramid(laplacian_a)
    
    # Clip values to valid range and convert to 8-bit image
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    return blended_image


if __name__ == "__main__":
    
    # Preparing...
    images_folder = 'images'
    hanoch_img = os.path.join(images_folder, 'hanoch.jpg')
    hanoch_resized_img = os.path.join(images_folder, 'hanoch_resized.jpg')
    monkey_img = os.path.join(images_folder, 'monkey.jpg')
    hanoch_mask_path = os.path.join(images_folder, 'hanoch_mask.jpg')
    mask_path = os.path.join(images_folder, 'mask.jpg')

    resize_img_according_another_img(hanoch_img, monkey_img, resized_image_path=hanoch_resized_img)
    make_non_black_white(image_path=hanoch_mask_path, output_path=mask_path)
    
    # Load two images and a mask
    image1 = cv2.imread(monkey_img)  # First image
    image2 = cv2.imread(hanoch_img)  # Second image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Mask (grayscale)

    # Blend the images
    blended = blend_images(image1, image2, mask, levels=5)

    # Show and save the result
    hanoch_as_monkey_path = "results/hanoch_as_monkey.jpg"
    cv2.imwrite(hanoch_as_monkey_path, blended)
    print(f"{hanoch_as_monkey_path} created successfully!")
