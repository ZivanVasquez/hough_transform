import numpy as np
import imageio
import math

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
    """
    Hough transform for lines

    Input:
    img - 2D binary image with nonzeros representing edges
    angle_step - Spacing between angles to use every n-th angle
                 between -90 and 90 degrees. Default step is 1.
    lines_are_white - boolean indicating whether lines to be detected are white
    value_threshold - Pixel values above or below the value_threshold are edges

    Returns:
    accumulator - 2D array of the hough transform accumulator
    theta - array of angles used in computation, in radians.
    rhos - array of rho values. Max size is 2 times the diagonal
           distance of the input image.
    """
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    # (row, col) indexes to edges
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    y_idxs, x_idxs = np.nonzero(are_edges)

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos

def fast_hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
    """hough line using vectorized numpy operations,
    may take more memory, but takes much less time"""
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step)) #can be changed
    #width, height = col.size  #if we use pillow
    width, height = img.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))   # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas))
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    #are_edges = cv2.Canny(img,50,150,apertureSize = 3)
    y_idxs, x_idxs = np.nonzero(are_edges)  # (row, col) indexes to edges
    # Vote in the hough accumulator
    xcosthetas = np.dot(x_idxs.reshape((-1,1)), cos_theta.reshape((1,-1)))
    ysinthetas = np.dot(y_idxs.reshape((-1,1)), sin_theta.reshape((1,-1)))
    rhosmat = np.round(xcosthetas + ysinthetas) + diag_len
    rhosmat = rhosmat.astype(np.int16)
    for i in range(num_thetas):
        rhos,counts = np.unique(rhosmat[:,i], return_counts=True)
        accumulator[rhos,i] = counts
    return accumulator, thetas, rhos

def show_hough_line(img, accumulator, thetas, rhos, save_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].axis('image')

    ax[1].imshow(
        accumulator, cmap='jet',
        extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    # plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def create_r_table(template, ref_point=None, angle_step=1):
    """
    Create R-Table (Reference Table) for Generalized Hough Transform
    
    Parameters:
    template - Binary template image of the shape to detect
    ref_point - Reference point (x, y). If None, uses center of template
    angle_step - Angle discretization step in degrees
    
    Returns:
    r_table - Dictionary where keys are angles and values are lists of (r, phi) pairs
    """
    if ref_point is None:
        # Use center of template as reference point
        ref_point = (template.shape[1] // 2, template.shape[0] // 2)
    
    # Find edge points in template
    y_edges, x_edges = np.where(template > 0)
    
    if len(x_edges) == 0:
        return {}
    
    # Create R-Table
    r_table = {}
    angles = np.arange(0, 360, angle_step)
    
    for angle in angles:
        r_table[angle] = []
    
    for x, y in zip(x_edges, y_edges):
        # Calculate vector from reference point to edge point
        dx = x - ref_point[0]
        dy = y - ref_point[1]
        
        # Calculate polar coordinates
        r = np.sqrt(dx**2 + dy**2)
        if r == 0:
            continue
        
        phi = np.arctan2(dy, dx)  # Angle in radians
        phi_deg = int(np.round(np.degrees(phi) % 360 / angle_step) * angle_step)
        
        r_table[phi_deg].append((r, phi))
    
    # Remove empty entries
    r_table = {k: v for k, v in r_table.items() if v}
    
    return r_table


def generalized_hough_transform(image, r_table, angle_step=1, scale_range=(0.5, 2.0), scale_step=0.1):
    """
    Generalized Hough Transform for arbitrary shape detection
    
    Parameters:
    image - Input image (grayscale or binary)
    r_table - Reference table from create_r_table
    angle_step - Angle discretization step
    scale_range - Tuple of (min_scale, max_scale)
    scale_step - Scale discretization step
    
    Returns:
    accumulator - 3D accumulator [y, x, scale]
    """
    if image.ndim == 3:
        image = rgb2gray(image)
    
    # Find edge points
    y_edges, x_edges = np.where(image > 0)
    
    if len(x_edges) == 0:
        return np.zeros((image.shape[0], image.shape[1], 1))
    
    # Create accumulator
    scales = np.arange(scale_range[0], scale_range[1] + scale_step, scale_step)
    accumulator = np.zeros((image.shape[0], image.shape[1], len(scales)))
    
    # Voting
    for x, y in zip(x_edges, y_edges):
        # Calculate gradient direction (approximate)
        # For simplicity, we'll use a simple approximation
        # In practice, you might want to use Sobel or Canny for better gradients
        
        # For each possible orientation in r_table
        for angle in r_table:
            for r, phi in r_table[angle]:
                for i, scale in enumerate(scales):
                    # Calculate potential reference point
                    dx = r * scale * np.cos(phi)
                    dy = r * scale * np.sin(phi)
                    
                    ref_x = int(x + dx)
                    ref_y = int(y + dy)
                    
                    # Check bounds
                    if (0 <= ref_x < image.shape[1] and 
                        0 <= ref_y < image.shape[0]):
                        accumulator[ref_y, ref_x, i] += 1
    
    return accumulator


def find_best_matches(accumulator, threshold=0.5, min_distance=10):
    """
    Find the best matches from the accumulator
    
    Parameters:
    accumulator - 3D accumulator from generalized_hough_transform
    threshold - Minimum vote threshold (fraction of max)
    min_distance - Minimum distance between detected shapes
    
    Returns:
    matches - List of (y, x, scale, votes) tuples
    """
    max_votes = accumulator.max()
    if max_votes == 0:
        return []
    
    # Find local maxima
    matches = []
    for y in range(accumulator.shape[0]):
        for x in range(accumulator.shape[1]):
            for s in range(accumulator.shape[2]):
                votes = accumulator[y, x, s]
                if votes > threshold * max_votes:
                    # Check if this is a local maximum
                    is_local_max = True
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            for ds in [-1, 0, 1]:
                                if dy == 0 and dx == 0 and ds == 0:
                                    continue
                                ny, nx, ns = y + dy, x + dx, s + ds
                                if (0 <= ny < accumulator.shape[0] and 
                                    0 <= nx < accumulator.shape[1] and
                                    0 <= ns < accumulator.shape[2]):
                                    if accumulator[ny, nx, ns] > votes:
                                        is_local_max = False
                                        break
                            if not is_local_max:
                                break
                        if not is_local_max:
                            break
                    
                    if is_local_max:
                        # Check minimum distance from existing matches
                        too_close = False
                        for my, mx, ms, mv in matches:
                            dist = np.sqrt((y - my)**2 + (x - mx)**2)
                            if dist < min_distance:
                                too_close = True
                                break
                        
                        if not too_close:
                            matches.append((y, x, s, votes))
    
    # Sort by votes descending
    matches.sort(key=lambda x: x[3], reverse=True)
    return matches


def show_ght_results(image, accumulator, matches, save_path=None):
    """
    Visualize Generalized Hough Transform results
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show original image with detected shapes
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Detected Shapes')
    
    # Mark detected shapes
    for y, x, scale_idx, votes in matches:
        ax1.plot(x, y, 'ro', markersize=8, alpha=0.7)
        # Draw circle around detection
        circle = plt.Circle((x, y), 5, color='red', fill=False, linewidth=2)
        ax1.add_artist(circle)
    
    # Show accumulator max projection
    max_accum = np.max(accumulator, axis=2)
    ax2.imshow(max_accum, cmap='hot')
    ax2.set_title('Accumulator (Max Projection)')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()


if __name__ == '__main__':
    # Example 1: Standard Hough Transform (lines)
    print("Running Standard Hough Transform...")
    imgpath = 'imgs/binary_crosses.png'
    img = imageio.imread(imgpath)
    if img.ndim == 3:
        img = rgb2gray(img)
    accumulator, thetas, rhos = hough_line(img)
    show_hough_line(img, accumulator, thetas, rhos, save_path='imgs/output.png')
    
    # Example 2: Generalized Hough Transform (arbitrary shapes)
    print("\nRunning Generalized Hough Transform...")
    
    # Create a simple template (e.g., a circle or cross)
    template = np.zeros((50, 50), dtype=np.uint8)
    # Draw a simple shape (cross)
    template[20:30, 24:26] = 255  # Vertical line
    template[24:26, 20:30] = 255  # Horizontal line
    
    # Create R-Table
    r_table = create_r_table(template)
    
    # Apply Generalized Hough Transform
    accumulator = generalized_hough_transform(img, r_table)
    matches = find_best_matches(accumulator)
    
    print(f"Found {len(matches)} shape matches")
    show_ght_results(img, accumulator, matches, save_path='imgs/ght_output.png')
