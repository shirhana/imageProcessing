from ex1 import main
import os

# Define the directory path
directory_path = 'Test/combined_takes'

# Get a list of all files in the directory
files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

output = """(28, 29)
(28, 29)
(35, 36)
(35, 36)
(57, 58)
(57, 58)
(71, 72)
(71, 72)
(31, 32)
(31, 32)
(35, 36)
(35, 36)
(90, 91)
(90, 91)
(124, 125)
(124, 125)
(37, 38)
(37, 38)
(30, 31)
(30, 31)
(33, 34)
(33, 34)
(9, 10)
(9, 10)""".splitlines()

index = 0
error_count = 0
for file in files:
    if output[index] != str(main(file,1)):
        print("Error, result for file:'",file,"' with type:",1,"should be:",output[index],"but yours is:",str(main(file,1)))
        error_count += 1
    index+=1
    if output[index] != str(main(file,2)):
        print("Error, result for file:'",file,"' with type:",2,"should be:",output[index],"but yours is:",str(main(file,2)))
        error_count += 1
    index+=1

print("You have",error_count,"Erros")



def extract_and_plot_frame(video_path, frame_number):
    import cv2
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open video.")
        return
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: Couldn't read the frame {frame_number}.")
        return
    
    # Convert frame from BGR to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Close the video capture
    cap.release()
    
    # Plot the extracted frame
    plt.imshow(frame_rgb)
    plt.axis('off')  # Hide axes
    plt.show()


# import os
# vs = [
#     # "videos/video1_category1.mp4",
#     # "videos/video2_category1.mp4",
#     # "videos/video3_category2.mp4",
#     # "videos/video4_category2.mp4",
#     "Test/combined_takes/combined_For_Bigger_Escape_take_2_4_Elephant_Dream_take_32_35.mp4"
# ]
# for v in vs:
#     # print(f"for {v}: {main(v, '')}")
#     extract_and_plot_frame(video_path=v, frame_number=16)