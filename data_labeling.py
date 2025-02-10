import os
import cv2
import numpy as np
import pandas as pd
from constants import *

def label_image_with_curves(image_path):
    name = image_path.split("/")[-1].split(".")[0]
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
    cv2.resizeWindow('Image', 1000, 800)
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

def show_points(image_path):
    df = pd.read_csv(POINTS_PATH) if os.path.exists(POINTS_PATH) else pd.DataFrame(columns=HEADER)
    image = cv2.imread(image_path) if os.path.exists(image_path) else None
    name = image_path.split("/")[-1].split(".")[0]
    points = df[df['name'] == name].iloc[0, 1:].tolist()
    for i in range(3):
        pts = np.array(points[i*10:(i+1)*10], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=False, color=(255, 0, 0), thickness=2)
    for i in range(0, 30, 2):
        cv2.circle(image, (points[i], points[i+1]), 3, (0, 255, 0), -1)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    if os.path.exists(POINTS_PATH):
        df = pd.read_csv(POINTS_PATH)
    else:
        df = pd.DataFrame(columns=HEADER)
    
    for i in os.listdir(DATA_PROCESSED_PATH):
        if not i.endswith(".jpg"): continue
        name = i.split(".")[0]
        if name in df['name'].values:
            continue

        result = label_image_with_curves(f"{DATA_PROCESSED_PATH}/{i}")
        if result == 'stop':
            break
        if result is None:
            continue
        
        new_points = frame_points(result, name)
        if name in df['name'].values:
            df[df['name'] == name].iloc[0,:] = new_points
        else:   
            df = pd.concat([df, new_points], ignore_index=True)

        df.to_csv(POINTS_PATH, index=False)
        print(f'{i} labeled')

if __name__ == "__main__":
    main()
