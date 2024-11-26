import cv2
import numpy as np

def transform_to_ellipse_space(ellipse):
    (cx, cy), (axes), angle = ellipse
    a, b = axes[0] / 2, axes[1] / 2  # Semi-major and semi-minor axes
    theta = np.radians(angle)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    def transform(point):
        x, y = point
        x_prime = (x - cx) * cos_theta + (y - cy) * sin_theta
        y_prime = -(x - cx) * sin_theta + (y - cy) * cos_theta
        return x_prime, y_prime
    def transform_back(point):
        x_prime, y_prime = point
        x = x_prime * cos_theta - y_prime * sin_theta + cx
        y = x_prime * sin_theta + y_prime * cos_theta + cy
        return x, y
    
    def find_tangent_slop(point):
        x_prime, y_prime = transform(point)
        if y_prime != 0:  # Avoid division by zero
            slope_tangent = -((b**2 / a**2) * x_prime) / y_prime
            angle_tangent = np.arctan(slope_tangent)
        else:
            slope_tangent = float('inf')  # Vertical tangent line
            angle_tangent = np.pi / 2


        new_point = (x_prime + np.array([100]), y_prime + np.array([100* slope_tangent]))
        return angle_tangent - theta, transform_back(new_point)

    return transform, transform_back, find_tangent_slop

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
        # Sort intersection points by the value of x
        intersection_points.sort(key=lambda point: point[0])

    return intersection_points

def surface_tension(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detector
    edges = cv2.Canny(gray, 80, 101)
    
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

    # Draw the contours
    if True:
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        cv2.imshow('Drop Outline and Surface', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
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

    # Create transformation functions for ellipse space
    transform, transform_back, find_tangent_angle = transform_to_ellipse_space(ellipse)
    
    # Draw the fitted line for the surface
    lefty = int((-x * vy / vx) + y)
    righty = int(((image.shape[1] - x) * vy / vx) + y)
    cv2.line(image, (image.shape[1] - 1, righty), (0, lefty), (255, 0, 0), 2)
    
    # Draw the contours
    # cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
    # cv2.drawContours(image, [drop_contour], -1, (0, 255, 0), 2)

    # Draw the fitted ellipse for the drop
    cv2.ellipse(image, ellipse, (0, 255, 0), 2)

    # Find intersection points of the line and the ellipse
    line = np.array([vx, vy, x, y])
    intersections = ellipse_line_intersection(line, ellipse)

    line_angel = np.arctan(vy/vx)

    if len(intersections) == 2:
        left_intersection, right_intersection = intersections
        
        # Draw intersection points
        cv2.circle(image, tuple(map(int, left_intersection)), 5, (0, 0, 255), -1)
        cv2.circle(image, tuple(map(int, right_intersection)), 5, (0, 0, 255), -1)

        # Calculate and draw tangent lines at the intersection points
        left_tangent_angel, left_tangent_point = find_tangent_angle(left_intersection)
        right_tangent_angel, right_tangent_point = find_tangent_angle(right_intersection)

        left_contact_angle = left_tangent_angel * -1
        right_contact_angle = right_tangent_angel + np.pi

        print('Left contact angle:', np.degrees(left_contact_angle)[0], 'degrees')
        print('Right contact angle:', np.degrees(right_contact_angle)[0], 'degrees')
        print('average contact angle:', np.degrees((left_contact_angle + right_contact_angle) / 2)[0], 'degrees')

        cv2.line(image, tuple(map(int, left_intersection)), tuple(map(int,left_tangent_point)), (0, 255, 255), 2)    
        cv2.line(image, tuple(map(int, right_intersection)), tuple(map(int, right_tangent_point)), (0, 255, 255), 2)


    # Add legend
    cv2.putText(image, 'Drop Contour', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, 'Surface Contour', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(image, 'Intersection Points', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(image, 'Tangent Lines', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Show the image
    cv2.imshow('Drop Outline and Surface', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


        

if __name__ == '__main__':
    image_path = r"C:\Users\yaniv\Yehonathan TAU\PhyChemLab\surface tension imges\copper.jpg"
    surface_tension(image_path)