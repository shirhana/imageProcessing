import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


DYNAMIC_VIDEO = "dynamic"
VIEWPOINT_VIDEO = "viewpoint"


class VideoMosaic:
    def __init__(self, first_image, init_offset, output_height_times=2, output_width_times=4, visualize=True):
        """This class processes every frame and generates the panorama"""
        ######## ADDED THESE LINES ########
        self.frame_width = first_image.shape[1]
        self.frame_height = first_image.shape[0]
        ###################################

        self.detector = cv2.SIFT_create(700)
        self.bf = cv2.BFMatcher()

        self.visualize = visualize

        self.process_first_frame(first_image)

        self.output_img = np.zeros(shape=(int(output_height_times * first_image.shape[0]), int(
            output_width_times * first_image.shape[1]), first_image.shape[2]))

        # offset
        self.w_offset = int(self.output_img.shape[0] / 2 - first_image.shape[0] / 2)
        self.h_offset = int(self.output_img.shape[1] / 2 - first_image.shape[1] / 2)
        self.h_offset = init_offset
        self.output_img[self.w_offset:self.w_offset + first_image.shape[0],
                        self.h_offset:self.h_offset + first_image.shape[1], :] = first_image

        self.H_old = np.eye(3)
        self.H_old[0, 2] = self.h_offset
        self.H_old[1, 2] = self.w_offset

        # Initialize canvas array to store all transformed frames
        self.canvas = []

    def process_first_frame(self, first_image):
        """processes the first frame for feature detection and description

        Args:
            first_image (cv2 image/np array): first image for feature detection
        """
        self.frame_prev = first_image
        frame_gray_prev = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        self.kp_prev, self.des_prev = self.detector.detectAndCompute(frame_gray_prev, None)

    def match(self, des_cur, des_prev):
        """matches the descriptors

        Args:
            des_cur (np array): current frame descriptor
            des_prev (np array): previous frame descriptor

        Returns:
            array: and array of matches between descriptors
        """
        # matching
        pair_matches = self.bf.knnMatch(des_cur, des_prev, k=2)
        matches = []
        for m, n in pair_matches:
            if m.distance < 0.7 * n.distance:
                matches.append(m)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # get the maximum of 20 best matches
        matches = matches[:min(len(matches), 20)]

        ##################### visualize #####################
        if self.visualize:
            match_img = cv2.drawMatches(
                self.frame_cur, self.kp_cur, self.frame_prev, self.kp_prev, matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            # Resize the match image to a smaller size for display
            scale_factor = 0.5  # Adjust this scale factor as needed (e.g., 0.3 for even smaller size)
            match_img_resized = cv2.resize(match_img, None, fx=scale_factor, fy=scale_factor)

            # Display the resized match image
            cv2.imshow('matches', match_img_resized)

        return matches

    def process_frame(self, frame_cur):
        """gets an image and processes that image for mosaicing

        Args:
            frame_cur (np array): input of current frame for the mosaicing
        """
        self.frame_cur = frame_cur
        frame_gray_cur = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY)
        self.kp_cur, self.des_cur = self.detector.detectAndCompute(frame_gray_cur, None)

        if self.visualize:
            # Visualize the keypoints on the current frame
            frame_with_keypoints = cv2.drawKeypoints(
                self.frame_cur, 
                self.kp_cur, 
                None, 
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            # Display the frame with keypoints
            cv2.imshow("Keypoints Visualization", frame_with_keypoints)

        self.matches = self.match(self.des_cur, self.des_prev)

        if len(self.matches) < 4:
            return

        self.H = self.find_rigid_transform(self.kp_cur, self.kp_prev, self.matches)
        # Extend the 2x3 affine matrix to 3x3
        H_extended = np.eye(3)
        H_extended[:2, :] = self.H

        # Multiply with the previous transformation matrix
        self.H = np.matmul(self.H_old, H_extended)
        # self.H = np.matmul(self.H_old, self.H)

        self.warp_and_store(self.frame_cur, self.H)

        # loop preparation
        self.H_old = self.H
        self.kp_prev = self.kp_cur
        self.des_prev = self.des_cur
        self.frame_prev = self.frame_cur

    @staticmethod
    def find_rigid_transform(image_1_kp, image_2_kp, matches):
        """
        Gets two matches and calculates the rigid transformation (rotation + translation) between two images.

        Args:
            image_1_kp (np array): Keypoints of image 1.
            image_2_kp (np array): Keypoints of image 2.
            matches (np array): Matches between keypoints in image 1 and image 2.

        Returns:
            np array of shape [2, 3]: Rigid transformation matrix.
        """
        image_1_points = np.zeros((len(matches), 2), dtype=np.float32)
        image_2_points = np.zeros((len(matches), 2), dtype=np.float32)
        for i in range(len(matches)):
            image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
            image_2_points[i] = image_2_kp[matches[i].trainIdx].pt

        transform, mask = cv2.estimateAffinePartial2D(image_1_points, image_2_points, method=cv2.RANSAC)

        return transform

    def warp_and_store(self, frame_cur, H):
        """
        Warps the current frame based on the calculated rigid transformation H and stores it in canvas.
        """
        # Use warpAffine instead of warpPerspective
        warped_img = cv2.warpAffine(
            frame_cur, H[:2], (self.output_img.shape[1], self.output_img.shape[0]), flags=cv2.INTER_LINEAR
        )

        # Store the warped frame in canvas
        self.canvas.append(warped_img)
        
        ##################### visualize #####################
        if self.visualize:
            scale_factor = 0.5
            warped_img_resized = cv2.resize(warped_img, None, fx=scale_factor, fy=scale_factor)
            cv2.imshow("Warped Frame", warped_img_resized)


def combine_images(img1, img2):
    """
    Combines two images, prioritizing color information from the first image.

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.

    Returns:
        np.ndarray: Combined image.
    """
    assert img1.shape == img2.shape, "Images must have the same dimensions"
    mask1 = np.any(img1 > 0, axis=-1)
    mask2 = np.any(img2 > 0, axis=-1)
    result = img1.copy()
    result[~mask1 & mask2] = img2[~mask1 & mask2]
    return result

def sort_corners(points):
    """
    Sorts corner coordinates in the order: top-left, bottom-left, top-right, bottom-right.

    Args:
        points (np.ndarray): Array of corner coordinates (4x2).

    Returns:
        np.ndarray: Sorted corner coordinates.
    """
    points = sorted(points, key=lambda x: x[0])  # Sort by x-coordinate
    left = points[:2]
    right = points[2:]
    left = sorted(left, key=lambda x: x[1])  # Sort left corners by y-coordinate
    right = sorted(right, key=lambda x: x[1])  # Sort right corners by y-coordinate
    return np.array([left[0], left[1], right[0], right[1]], dtype=np.float32)

def detect_rectangle_corners(image):
    """
    Detects the corners of a rectangular region in a padded image.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Array of detected corner coordinates (4x2).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
     # Ensure approx has 4 points
    if len(approx) > 4:
        approx = approx[:4]
    return sort_corners(approx.reshape(4, 2))

def extract_region(image, corners, null_right, null_left):
    """
    Extracts a portion of the image within a specified region defined by the corner coordinates and null values.

    Args:
        image (np.ndarray): Input image.
        corners (np.ndarray): Array of corner coordinates (4x2).
        null_right (float): Proportion of the image to exclude from the right side.
        null_left (float): Proportion of the image to exclude from the left side.

    Returns:
        np.ndarray: Extracted portion of the image.
    """
    # Extract corner coordinates
    x1, y1 = corners[0]  # Top-left corner
    x2, y2 = corners[1]  # Bottom-left corner
    x3, y3 = corners[2]  # Top-right corner
    x4, y4 = corners[3]  # Bottom-right corner
    top_line = lambda x: ((y3 - y1) / (x3 - x1)) * (x - x1) + y1
    bottom_line = lambda x: ((y4 - y2) / (x4 - x2)) * (x - x2) + y2

    top_left_x = x1 + (x3 - x1) * (1 - null_right)
    top_left_y = top_line(top_left_x)
    top_left = (top_left_x, top_left_y)
    bottom_left_x = x2 + (x4 - x2) * (1 - null_right)
    bottom_left_y = bottom_line(bottom_left_x)
    bottom_left = (bottom_left_x, bottom_left_y)
    top_right_x = x3 - (x3 - x1) * (1 - null_left)
    top_right_y = top_line(top_right_x)
    top_right = (top_right_x, top_right_y)
    bottom_right_x = x4 - (x4 - x2) * (1 - null_left)
    bottom_right_y = bottom_line(bottom_right_x)
    bottom_right = (bottom_right_x, bottom_right_y)

    # Create mask 
    mask = np.zeros_like(image, dtype=np.uint8)
    p = np.array([top_left, bottom_left, bottom_right, top_right], dtype=np.int32)
    mask = cv2.fillPoly(mask, [p], (255, 255, 255))
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

def visualize_panoramas(canvases, num_frames, canvas_size=(400, 400)):
    """
    Visualizes the panoramas side by side, highlighting the regions of focus for each one.

    Args:
        canvases (list): List of panoramic images.
        num_frames (int): Number of frames to display.
        canvas_size (tuple): Size to resize individual panoramas for visualization.
    """
    visualizations = []

    for i in range(num_frames):
        canvas = canvases[i]
        resized_canvas = cv2.resize(canvas, canvas_size)

        # Highlight regions of focus
        height, width = resized_canvas.shape[:2]
        left_focus = int(width * (i / num_frames))
        right_focus = int(width * ((i + 1) / num_frames))

        # Add semi-transparent overlay
        overlay = resized_canvas.copy()
        cv2.rectangle(
            overlay, 
            (left_focus, 0), 
            (right_focus, height), 
            color=(0, 255, 0), 
            thickness=-1
        )
        alpha = 0.3
        highlighted_canvas = cv2.addWeighted(overlay, alpha, resized_canvas, 1 - alpha, 0)
        visualizations.append(highlighted_canvas)

    # Combine the visualizations horizontally
    combined_visualization = np.hstack(visualizations)
    cv2.imshow("Panoramas with Focus Regions", combined_visualization)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()


def execute_for_viewpoint(video_mosaic, canvases, num_frames):
    for i in range(num_frames):
        # Find range of frames that have at least 500 non-zero pixels in the current portion of the canvas
        flag = False
        last_bottom_left_x = None
        for j in range(0,len(video_mosaic.canvas),5):
            # Count non-zero pixels in the current range
            corners = detect_rectangle_corners(video_mosaic.canvas[j])
            height_corners = corners[1][1] - corners[0][1]
            width_corners = corners[2][0] - corners[0][0]
            bottom_left_x = corners[1][0]
            if last_bottom_left_x is not None and (last_bottom_left_x - bottom_left_x) < 20:
                continue
            pixel_count = height_corners * width_corners * 0.7 * (1/num_frames)
            null_right = (i)*(1/num_frames)
            null_left = (num_frames-1-i)*(1/num_frames)
            window = extract_region(video_mosaic.canvas[j], corners, null_right*0.8, null_left*0.8)
            non_zero_count = np.count_nonzero(window)
            
            if non_zero_count < pixel_count:
                continue

            if (non_zero_count < pixel_count and flag) or i == num_frames:
                break

            flag = True

        # Assign to canvases if the condition is met
            canvases[i] = combine_images(canvases[i], window)

    # Visualize the panoramas after processing
    # visualize_panoramas(canvases, num_frames)

    return canvases


def visualize_dynamic_mosaic(final_mosaic, focus_mask, excluded_mask):
    """
    Visualizes the final mosaic with overlays or heatmaps to highlight selected and excluded areas.

    Args:
        final_mosaic (np.ndarray): Final mosaic image.
        focus_mask (np.ndarray): Binary mask highlighting selected areas.
        excluded_mask (np.ndarray): Binary mask highlighting excluded areas.
    """
    # Create a heatmap for focus regions
    focus_overlay = final_mosaic.copy()
    focus_overlay[focus_mask > 0] = (0, 255, 0)  # Highlight selected areas in green

    # Create a heatmap for excluded regions
    excluded_overlay = final_mosaic.copy()
    excluded_overlay[excluded_mask > 0] = (0, 0, 255)  # Highlight excluded areas in red

    # Plot the results
    plt.figure(figsize=(15, 10))

    plt.subplot(1, 3, 1)
    plt.title("Final Mosaic")
    plt.imshow(cv2.cvtColor(final_mosaic, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Focus Areas (Green)")
    plt.imshow(cv2.cvtColor(focus_overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Excluded Areas (Red)")
    plt.imshow(cv2.cvtColor(excluded_overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()



def execute_for_dynamic(video_mosaic, canvases, num_frames):
    pixel_count = 0.6*video_mosaic.frame_height*20
    # Create masks for visualization
    focus_mask = np.zeros_like(canvases[0][:, :, 0], dtype=np.uint8)
    excluded_mask = np.zeros_like(canvases[0][:, :, 0], dtype=np.uint8)

    for i in range(0, canvases[0].shape[1], 20):
        # Find range of frames that have at least 500 non-zero pixels in the current portion of the canvas
        start = i
        end = i + 20
        counter = 0
        flag = False
        for j in range(0, len(video_mosaic.canvas), 3):
            # Count non-zero pixels in the current range
            non_zero_count = np.count_nonzero(video_mosaic.canvas[j][:, start:end, :])
            
            if non_zero_count < pixel_count:
                continue

            if (non_zero_count < pixel_count and flag) or counter == num_frames:
                break

            flag = True
            focus_mask[:, start:end] = 255
            excluded_mask[:, start:end] = 0

            # Assign to canvases if the condition is met
            canvases[counter][:, start:end, :] = video_mosaic.canvas[j][:, start:end, :]
            counter += 1

    # Combine all canvases into a single final mosaic
    final_mosaic = np.zeros_like(canvases[0])
    for canvas in canvases:
        final_mosaic = combine_images(final_mosaic, canvas)

    # Visualize the final mosaic with focus and excluded regions
    # visualize_dynamic_mosaic(final_mosaic, focus_mask, excluded_mask)

    return canvases


def create_images_per_frames(video_path, images_folder, final_image_prefix, execution_type, init_offset, visualize):
    cap = cv2.VideoCapture(video_path)
    is_first_frame = True
    video_mosaic = None

    while cap.isOpened():
        ret, frame_cur = cap.read()
        if not ret:
            if is_first_frame:
                continue
            break

        if is_first_frame:
            video_mosaic = VideoMosaic(frame_cur, init_offset=init_offset, visualize=visualize)
            is_first_frame = False
            continue

        # Process each frame
        video_mosaic.process_frame(frame_cur)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    num_frames = 30
    canvases = [np.zeros_like(video_mosaic.canvas[0]) for _ in range(num_frames)]

    if execution_type == DYNAMIC_VIDEO:
        canvases = execute_for_dynamic(video_mosaic, canvases, num_frames)
    else: 
        canvases = execute_for_viewpoint(video_mosaic, canvases, num_frames)

    # Make canvases a video
    for i in range(len(canvases)):
        canvases[i] = cv2.resize(canvases[i], (canvases[i].shape[1] // 5, canvases[i].shape[0] // 5))
        cv2.imwrite(f'{images_folder}/{final_image_prefix}{i}.jpg', canvases[i])
    cv2.destroyAllWindows()

def create_video(images_prefix, image_folder, output_video_name, fps=10):
    # Get the list of image files in sorted order
    images = [img for img in os.listdir(image_folder) if img.startswith(images_prefix) and img.endswith('.jpg')]
    images.sort(key=lambda x: int(x.split(images_prefix)[1].split('.')[0]))  # Sort by the numerical part of the filename

    # Read the first image to determine the frame size
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)

    # ensure all dimensions are the same
    for i in range(1, len(images)):
        image_path = os.path.join(image_folder, images[i])
        frame = cv2.imread(image_path)
        assert frame.shape == first_image.shape, "Images must have the same dimensions"

    # Initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_video_name, fourcc, fps, (first_image.shape[1], first_image.shape[0]))

    # Apply the mask and write cropped frames to the video
    for i in range(0, len(images)):
        image_path = os.path.join(image_folder, images[i])

        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the VideoWriter
    video_writer.release()


def execute_stereo_mosaicing(video_input_file, video_output_file, images_folder, image_prefix, init_offset, execution_type=DYNAMIC_VIDEO, visualize=False):
    create_images_per_frames(video_path=video_input_file, images_folder=images_folder, final_image_prefix=image_prefix, execution_type=execution_type, init_offset=init_offset, visualize=visualize)
    create_video(images_prefix=image_prefix, image_folder=images_folder, output_video_name=video_output_file)
    print(f"{video_output_file} created successfully!\nExecution Type: {execution_type}")


if __name__ == "__main__":
    images_folder = "test"
    os.makedirs(images_folder, exist_ok=True)

    execute_stereo_mosaicing(
        video_input_file="inputs/Iguazu.mp4", 
        video_output_file=f"outputs/Iguazu_output.mp4", 
        images_folder=images_folder, 
        init_offset=20, # this video is from left to right so init with 20 as offset
        image_prefix="mosaic_iguazu_", 
        execution_type=DYNAMIC_VIDEO,
        visualize=True # decide if want to debug with visualization in RUN TIME
    )
    execute_stereo_mosaicing(
        video_input_file="inputs/Kessaria.mp4", 
        video_output_file=f"outputs/Kessaria_output.mp4", 
        images_folder=images_folder, 
        init_offset=3000, # this video is from right to left so init with 3000 as offset
        image_prefix="mosaic_kessaria_", 
        execution_type=VIEWPOINT_VIDEO,
        visualize=True # decide if want to debug with visualization in RUN TIME
    )
    