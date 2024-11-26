import cv2
import numpy as np

def show_drop_and_surface(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detector
    edges = cv2.Canny(gray, 90, 100)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assume the largest contour includes both the surface edge and the drop
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Find the leftmost and rightmost points of the largest contour (surface edge)
    leftmost_points = largest_contour[np.argsort(largest_contour[:, :, 0].flatten())[:40]]
    rightmost_points = largest_contour[np.argsort(largest_contour[:, :, 0].flatten())[-40:]]
    
    # Combine the leftmost and rightmost points
    surface_points = np.vstack((leftmost_points, rightmost_points))
    
    # Fit a line to the surface edge
    [vx, vy, x, y] = cv2.fitLine(surface_points, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # Define delta value
    delta_value = 4200
    
    # Calculate the y-values of the surface line at each x-coordinate of the contour
    surface_y_values = (vx / vy) * (largest_contour[:, :, 1] - x) + y
    
    # Calculate the difference between the contour y-values and the surface y-values
    differences = np.abs(largest_contour[:, :, 1] - surface_y_values)
    
    # Extract the drop contour
    drop_contour = largest_contour[differences > delta_value]

    # Fit an ellipse to the drop contour
    ellipse = cv2.fitEllipse(drop_contour)
    
    
    # Draw the fitted line for the surface
    lefty = int((-x * vy / vx) + y)
    righty = int(((image.shape[1] - x) * vy / vx) + y)
    cv2.line(image, (image.shape[1] - 1, righty), (0, lefty), (255, 0, 0), 2)
    
    # Draw the contours
    # cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
    # cv2.drawContours(image, [drop_contour], -1, (0, 255, 0), 2)

    # Draw the fitted ellipse for the drop
    cv2.ellipse(image, ellipse, (0, 255, 0), 2)

    # Add legend
    cv2.putText(image, 'Drop Contour', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, 'Surface Contour', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Show the image
    cv2.imshow('Drop Outline and Surface', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = r"C:\Users\yaniv\Yehonathan TAU\PhyChemLab\surface tension imges\teflon3.jpg"
show_drop_and_surface(image_path)