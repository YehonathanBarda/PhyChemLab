import cv2
import numpy as np
import math

def ellipse_line_intersection(line, ellipse):
    # Line parameters
    vx, vy, x0, y0 = line[0], line[1], line[2], line[3]

    # Ellipse parameters
    (cx, cy), (axes), angle = ellipse
    a, b = axes[0] / 2, axes[1] / 2  # Semi-major and semi-minor axes
    theta = np.radians(angle)

    # Transformation coefficients
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    # Quadratic equation coefficients
    A = ((vx * cos_theta + vy * sin_theta)**2) / a**2 + ((vx * sin_theta - vy * cos_theta)**2) / b**2
    B = 2 * ((vx * cos_theta + vy * sin_theta) * ((x0 - cx) * cos_theta + (y0 - cy) * sin_theta) / a**2 +
             (vx * sin_theta - vy * cos_theta) * ((x0 - cx) * sin_theta - (y0 - cy) * cos_theta) / b**2)
    C = (((x0 - cx) * cos_theta + (y0 - cy) * sin_theta)**2) / a**2 + \
        (((x0 - cx) * sin_theta - (y0 - cy) * cos_theta)**2) / b**2 - 1

    # Solve the quadratic equation: At^2 + Bt + C = 0
    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        return []  # No intersection

    # Compute solutions for t
    t1 = (-B + np.sqrt(discriminant)) / (2 * A)
    t2 = (-B - np.sqrt(discriminant)) / (2 * A)

    # Back-substitute to find intersection points
    intersection_points = []
    for t in [t1, t2]:
        x = x0 + t * vx
        y = y0 + t * vy
        intersection_points.append((x, y))

    return intersection_points

def draw_tangent(image, ellipse, point, color):
    center, axes, angle = ellipse
    center = np.array(center)
    axes = np.array(axes) / 2
    angle = np.deg2rad(angle)
    
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    def transform(point):
        return np.array([
            cos_angle * (point[0] - center[0]) + sin_angle * (point[1] - center[1]),
            -sin_angle * (point[0] - center[0]) + cos_angle * (point[1] - center[1])
        ]) / axes
    
    transformed_point = transform(point)
    tangent_slope = -transformed_point[0] / transformed_point[1]
    tangent_angle = np.arctan(tangent_slope)
    
    length = 100
    dx = length * np.cos(tangent_angle)
    dy = length * np.sin(tangent_angle)
    
    pt1 = (int(point[0] - dx), int(point[1] - dy))
    pt2 = (int(point[0] + dx), int(point[1] + dy))
    
    cv2.line(image, pt1, pt2, color, 2)
    
    return tangent_angle

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
    
    # Draw the fitted ellipse for the drop
    cv2.ellipse(image, ellipse, (0, 255, 0), 2)
    
    # Find intersection points of the line and the ellipse
    line = np.array([vx, vy, x, y])
    intersections = ellipse_line_intersection(line, ellipse)
    
    if len(intersections) == 2:
        left_intersection, right_intersection = intersections
        
        # Draw intersection points
        cv2.circle(image, tuple(map(int, left_intersection)), 5, (0, 0, 255), -1)
        cv2.circle(image, tuple(map(int, right_intersection)), 5, (0, 0, 255), -1)
        
        # Calculate and draw tangent lines at the intersection points
        # left_tangent_angle = draw_tangent(image, ellipse, left_intersection, (0, 255, 255))
        # right_tangent_angle = draw_tangent(image, ellipse, right_intersection, (0, 255, 255))
        
        # Calculate contact angles
        # surface_angle = np.arctan2(vy, vx)
        # left_contact_angle = np.abs(surface_angle - left_tangent_angle)
        # right_contact_angle = np.abs(surface_angle - right_tangent_angle)
        
        # left_contact_angle_deg = np.degrees(left_contact_angle)
        # right_contact_angle_deg = np.degrees(right_contact_angle)
        
        # print(f"Left Contact Angle: {left_contact_angle_deg:.2f} degrees")
        # print(f"Right Contact Angle: {right_contact_angle_deg:.2f} degrees")
    
    # Add legend
    cv2.putText(image, 'Drop Contour', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, 'Surface Contour', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(image, 'Intersection Points', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(image, 'Tangent Lines', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Show the image
    cv2.imshow('Drop Outline and Surface', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = r"C:\Users\yaniv\Yehonathan TAU\PhyChemLab\surface tension imges\teflon3.jpg"
show_drop_and_surface(image_path)