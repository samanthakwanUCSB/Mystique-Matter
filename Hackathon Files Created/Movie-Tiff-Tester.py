import cv2
import tifffile
import numpy as np
import os

def movie_to_tif(video_path, output_tif_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return


    frames = []
    input_Value = input('Enter Yes if you want to convert every frame of your movie into the .tif stack. Enter No if you only want some of the frames to be converted.')
    
    if input_Value == 'Yes':
        #print("first if")
        x = 1
    else: 
        #print("else check")
        x = input("Enter how many frames you want to convert to the .tif stack (e.g. 2 is every 2 frames, 1 is every frame, etc.): ") 

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert color BGR frame to Grayscale 
        # Remove this line if you want to keep colors (will require RGB format)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)

    cap.release()

    # Convert the list of frames into one big NumPy array
    # The shape will be (Number_of_frames, Height, Width)
    stacked_frames = np.array(frames)

    #Choose a certain number of frames with the "x" value in the [::x] slot, x being 
    #the number frame you get (e.g. if x = 1 then you get every frame, x = 2 is every other frame)
    kept_frames = stacked_frames[::int(x)]

    # Save as multi-page TIFF
    tifffile.imwrite(output_tif_path, kept_frames)
    print(f"Success! Saved {len(kept_frames)} frames to {output_tif_path}")

# Example Usage
movie_Name = input('Enter the full path of your.mov file:')
movie_path  = movie_Name
#movie_path = "BLOOP Screen Recording 2026-06-24 at 9.34.52 PM.mov"
output_path = f'{movie_path}.tif'
movie_to_tif(movie_path, output_path)
