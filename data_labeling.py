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
            cv2.destroyAllWindows()

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 800, 600)
    cv2.setMouseCallback('Image', click_event)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return points if len(points) == 15 else None

def frame_points(points, name):
    points_array = np.array(points).flatten()
    points_array = pd.DataFrame([[name] + points_array.tolist()], columns=HEADER)
    return points_array

def main():
    if os.path.exists("points.csv"):
        df = pd.read_csv("points.csv", )
    else:
        df = pd.DataFrame(columns=HEADER)
    for i in ['FEMALE_50', 'FEMALE_51']:
        new_points = label_image_with_curves(f"./data_processed/{i}.jpg")
        new_points = frame_points(new_points, f"{i}")
        df = pd.concat([df, new_points], ignore_index=True)
    df.to_csv("points.csv", index=False)

if __name__ == "__main__":
    main()
