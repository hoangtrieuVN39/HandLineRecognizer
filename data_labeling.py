import os
import cv2
import numpy as np
import pandas as pd
from constants import *

def label_image_with_curves(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None  # Return None if the image cannot be loaded

    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Store the point
            points.append((x, y))
            # Draw a small circle at the point
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow('Image', image)

        if len(points) == 15:
            for i in range(3):
                # Convert points to a numpy array
                pts = np.array(points[i*5:(i+1)*5], np.int32)
                pts = pts.reshape((-1, 1, 2))
                # Draw the curve
                cv2.polylines(image, [pts], isClosed=False, color=(255, 0, 0), thickness=2)
            cv2.imshow('Image', image)
    
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 1000, 800)
    cv2.setMouseCallback('Image', click_event)
    cv2.imshow('Image', image)


    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to end labeling for this image
            break
        elif key == ord('w'):  # Press 'w' to end labeling for all images
            cv2.destroyAllWindows()
            return 'stop'
        elif key == ord('e'):  # Press 'e' to reset points
            points.clear()
            image = cv2.imread(image_path)  # Reload the image to clear drawn points
            cv2.imshow('Image', image)

    cv2.destroyAllWindows()

    return points if len(points) == 15 else None

def frame_points(points, filename):
    name = filename.split(".")[0]
    points_array = np.array(points).flatten()
    points_array = pd.DataFrame([[name] + points_array.tolist()], columns=HEADER)
    return points_array

def main():
    if os.path.exists("points.csv"):
        df = pd.read_csv("points.csv")
    else:
        df = pd.DataFrame(columns=HEADER)
    
    for i in os.listdir(DATA_PROCESSED_PATH):
        if not i.endswith(".jpg"): continue
        
        result = label_image_with_curves(f"{DATA_PROCESSED_PATH}/{i}")
        if result == 'stop':
            break
        if result is None:
            continue
        
        name = i.split(".")[0]
        new_points = frame_points(result, f"{i}")
        if name in df['name'].values:
            df[df['name'] == name].iloc[0,:] = new_points
        else:   
            df = pd.concat([df, new_points], ignore_index=True)

        df.to_csv("points.csv", index=False)

if __name__ == "__main__":
    main()
