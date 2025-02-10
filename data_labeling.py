import os
import cv2
import numpy as np
import pandas as pd
from constants import *

def label_image_with_curves(image_path, name):

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow('Image', image)

        if len(points) == 15:
            for i in range(3):

                pts = np.array(points[i*5:(i+1)*5], np.int32)
                pts = pts.reshape((-1, 1, 2))

                cv2.polylines(image, [pts], isClosed=False, color=(255, 0, 0), thickness=2)
            
            cv2.imshow('Image', image)
    
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 800, 600)
    cv2.setMouseCallback('Image', click_event)
    cv2.putText(image, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
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
            image = cv2.imread(image_path)
            cv2.putText(image, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow('Image', image)

    cv2.destroyAllWindows()

    return points if len(points) == 15 else None

def frame_points(points, name):
    points_array = np.array(points).flatten()
    points_array = pd.DataFrame([[name] + points_array.tolist()], columns=HEADER)
    return points_array

def main():
    list_of_names = []
    if os.path.exists("points.csv"):
        df = pd.read_csv("points.csv")
        list_of_names = df.iloc[:, 0].tolist()
    else:
        df = pd.DataFrame(columns=HEADER)
    
    for i in os.listdir(DATA_PROCESSED_PATH):
        if not i.endswith(".jpg"): continue
        name = i.split(".")[0]
        
        if name in list_of_names :
            continue
        result = label_image_with_curves(f"{DATA_PROCESSED_PATH}/{i}", name)
        if result == 'stop':
            break
        if result is None:
            continue
        
        new_points = frame_points(result, name)
        if name in df['name'].values:
            df[df['name'] == name].iloc[0,:] = new_points
        else:   
            df = pd.concat([df, new_points], ignore_index=True)

        df.to_csv("points.csv", index=False)
        print(f'{i} labeled')

if __name__ == "__main__":
    main()
