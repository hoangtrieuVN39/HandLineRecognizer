import cv2
import numpy as np

def label_image_with_curves(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Store the point
            points.append((x, y))
            # Draw a small circle at the point
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow('Image', image)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Draw the curve when right mouse button is clicked
            if len(points) > 1:
                # Convert points to a numpy array
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                # Draw the curve
                cv2.polylines(image, [pts], isClosed=False, color=(255, 0, 0), thickness=2)
                cv2.imshow('Image', image)

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 800, 600)
    cv2.setMouseCallback('Image', click_event)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    label_image_with_curves(r"E:/Máy tính/LinesOfTheHandRecognizer/data_processed/FEMALE_50.jpg")

if __name__ == "__main__":
    main()
