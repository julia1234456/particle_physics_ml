
import math 
import numpy as np
from CONSTANTS  import *

def normalize(histo, multi=255):
    """
    Normalize picture in [0,multi] range, with integer steps. E.g. multi=255 for 256 steps.
    """
    return (histo / np.max(histo) * multi).astype(int)

    
def pad_image(image, max_size=(MAX_HEIGHT, MAX_WIDTH)):
    """
    Simply pad an image with zeros up to max_size.
    """
    size = np.shape(image)
    px = max_size[0] - size[0]
    py = max_size[1] - size[1]
    image = np.pad(
        image,
        (
            list(map(int, (np.floor(px * 0.5), np.ceil(px * 0.5)))),
            list(map(int, (np.floor(py * 0.5), np.ceil(py * 0.5))))
        ),
        'constant'
    )
    return image

def mod_pi(angle):
    pi_p =  math.pi
    pi_n = -math.pi

    two_pi_p = 2 * pi_p
    two_pi_n = 2 * pi_n

    if angle > pi_p:
        angle += two_pi_n
        while angle > pi_p:
            angle += two_pi_n

    elif angle < pi_n:
        angle += two_pi_p
        while angle < pi_n:
            angle += two_pi_p

    return angle





def consts_to_image(consts, xedges, yedges):
    """
    consts: n x 3 np array, each row is a pixel, column 0 is the constituent pT (intensity), column 1 is the x coordinate, column 2 is the y coordinate
    xedges: *evenly spaced* array of numbers mapping the left x position of every pixel (+ right x position of the right-most pixel)
    yedges: *evenly spaced* array of numbers mapping the bottom y position of every pixel (+ top y position of the top-most pixel)

    return:
    2D np array with width = len(xedges) - 1 and height = len(yedges) - 1
    """
    width  = len(xedges) - 1
    height = len(yedges) - 1

    if width <= 0 or height <= 0:
        raise RuntimeError

    image = np.zeros((height, width))

    xedges = sorted(xedges)
    yedges = sorted(yedges, reverse=True)

    x_size = xedges[1] - xedges[0]
    y_size = yedges[0] - yedges[1]

    assert all(math.isclose(xedges[i + 1] - xedges[i], x_size) for i in range(1, width))
    assert all(math.isclose(yedges[i] - yedges[i + 1], y_size) for i in range(1, height))

    for pix, old_x, old_y in consts:
        # if old_x not in xedges or old_x == xedges[0] or old_x == xedges[-1] or \
        #    old_y not in yedges or old_y == yedges[0] or old_y == yedges[-1]:
        #     continue

        for x in range(width):
            if xedges[x] <= old_x <= xedges[x + 1]:
                break
        else:
            continue

        for y in range(height):
            if yedges[y] >= old_y >= yedges[y + 1]:
                break
        else:
            continue

        pix_min_x = xedges[x]
        pix_max_x = xedges[x + 1]
        pix_min_y = yedges[y + 1]
        pix_max_y = yedges[y]

        assert pix_max_x > pix_min_x and pix_max_y > pix_min_y

        same_pixel_x_center = (pix_min_x + pix_max_x) * 0.5
        same_pixel_y_center = (pix_min_y + pix_max_y) * 0.5

        # Impossible optimization
        if same_pixel_x_center == old_x and same_pixel_y_center == old_y:
            image[y, x] += pix
            continue

        # ratio_total = 0

        min_x = old_x - x_size * 0.5
        max_x = min_x + x_size
        min_y = old_y - y_size * 0.5
        max_y = min_y + y_size

        assert max_x > min_x and max_y > min_y

        area = x_size * y_size

        overlap_min_x = max(min_x, pix_min_x)
        overlap_max_x = min(max_x, pix_max_x)
        overlap_min_y = max(min_y, pix_min_y)
        overlap_max_y = min(max_y, pix_max_y)

        if overlap_max_x > overlap_min_x and overlap_max_y > overlap_min_y:
            overlap_area = (overlap_max_x - overlap_min_x) * (overlap_max_y - overlap_min_y)
            # assert 0 < overlap_area < area
            # ratio = overlap_area / area
            # ratio_total += ratio
            # image[y, x] += pix * ratio
            image[y, x] += pix * (overlap_area / area)

        # Top Left
        if x - 1 >= 0 and y - 1 >= 0:
            x -= 1
            y -= 1

            pix_min_x = xedges[x]
            pix_max_x = xedges[x + 1]
            pix_min_y = yedges[y + 1]
            pix_max_y = yedges[y]

            assert pix_max_x > pix_min_x and pix_max_y > pix_min_y

            overlap_min_x = max(min_x, pix_min_x)
            overlap_max_x = min(max_x, pix_max_x)
            overlap_min_y = max(min_y, pix_min_y)
            overlap_max_y = min(max_y, pix_max_y)

            if overlap_max_x > overlap_min_x and overlap_max_y > overlap_min_y:
                overlap_area = (overlap_max_x - overlap_min_x) * (overlap_max_y - overlap_min_y)
                # assert 0 < overlap_area < area
                # ratio = overlap_area / area
                # ratio_total += ratio
                # image[y, x] += pix * ratio
                image[y, x] += pix * (overlap_area / area)

            x += 1
            y += 1

        # Left
        if x - 1 >= 0:
            x -= 1

            pix_min_x = xedges[x]
            pix_max_x = xedges[x + 1]
            pix_min_y = yedges[y + 1]
            pix_max_y = yedges[y]

            assert pix_max_x > pix_min_x and pix_max_y > pix_min_y

            overlap_min_x = max(min_x, pix_min_x)
            overlap_max_x = min(max_x, pix_max_x)
            overlap_min_y = max(min_y, pix_min_y)
            overlap_max_y = min(max_y, pix_max_y)

            if overlap_max_x > overlap_min_x and overlap_max_y > overlap_min_y:
                overlap_area = (overlap_max_x - overlap_min_x) * (overlap_max_y - overlap_min_y)
                # assert 0 < overlap_area < area
                # ratio = overlap_area / area
                # ratio_total += ratio
                # image[y, x] += pix * ratio
                image[y, x] += pix * (overlap_area / area)

            x += 1

        # Bottom Left
        if x - 1 >= 0 and y + 1 < height:
            x -= 1
            y += 1

            pix_min_x = xedges[x]
            pix_max_x = xedges[x + 1]
            pix_min_y = yedges[y + 1]
            pix_max_y = yedges[y]

            assert pix_max_x > pix_min_x and pix_max_y > pix_min_y

            overlap_min_x = max(min_x, pix_min_x)
            overlap_max_x = min(max_x, pix_max_x)
            overlap_min_y = max(min_y, pix_min_y)
            overlap_max_y = min(max_y, pix_max_y)

            if overlap_max_x > overlap_min_x and overlap_max_y > overlap_min_y:
                overlap_area = (overlap_max_x - overlap_min_x) * (overlap_max_y - overlap_min_y)
                # assert 0 < overlap_area < area
                # ratio = overlap_area / area
                # ratio_total += ratio
                # image[y, x] += pix * ratio
                image[y, x] += pix * (overlap_area / area)

            x += 1
            y -= 1

        # Top Right
        if x + 1 < width and y - 1 >= 0:
            x += 1
            y -= 1

            pix_min_x = xedges[x]
            pix_max_x = xedges[x + 1]
            pix_min_y = yedges[y + 1]
            pix_max_y = yedges[y]

            assert pix_max_x > pix_min_x and pix_max_y > pix_min_y

            overlap_min_x = max(min_x, pix_min_x)
            overlap_max_x = min(max_x, pix_max_x)
            overlap_min_y = max(min_y, pix_min_y)
            overlap_max_y = min(max_y, pix_max_y)

            if overlap_max_x > overlap_min_x and overlap_max_y > overlap_min_y:
                overlap_area = (overlap_max_x - overlap_min_x) * (overlap_max_y - overlap_min_y)
                # assert 0 < overlap_area < area
                # ratio = overlap_area / area
                # ratio_total += ratio
                # image[y, x] += pix * ratio
                image[y, x] += pix * (overlap_area / area)

            x -= 1
            y += 1

        # Right
        if x + 1 < width:
            x += 1

            pix_min_x = xedges[x]
            pix_max_x = xedges[x + 1]
            pix_min_y = yedges[y + 1]
            pix_max_y = yedges[y]

            assert pix_max_x > pix_min_x and pix_max_y > pix_min_y

            overlap_min_x = max(min_x, pix_min_x)
            overlap_max_x = min(max_x, pix_max_x)
            overlap_min_y = max(min_y, pix_min_y)
            overlap_max_y = min(max_y, pix_max_y)

            if overlap_max_x > overlap_min_x and overlap_max_y > overlap_min_y:
                overlap_area = (overlap_max_x - overlap_min_x) * (overlap_max_y - overlap_min_y)
                # assert 0 < overlap_area < area
                # ratio = overlap_area / area
                # ratio_total += ratio
                # image[y, x] += pix * ratio
                image[y, x] += pix * (overlap_area / area)

            x -= 1

        # Bottom Right
        if x + 1 < width and y + 1 < height:
            x += 1
            y += 1

            pix_min_x = xedges[x]
            pix_max_x = xedges[x + 1]
            pix_min_y = yedges[y + 1]
            pix_max_y = yedges[y]

            assert pix_max_x > pix_min_x and pix_max_y > pix_min_y

            overlap_min_x = max(min_x, pix_min_x)
            overlap_max_x = min(max_x, pix_max_x)
            overlap_min_y = max(min_y, pix_min_y)
            overlap_max_y = min(max_y, pix_max_y)

            if overlap_max_x > overlap_min_x and overlap_max_y > overlap_min_y:
                overlap_area = (overlap_max_x - overlap_min_x) * (overlap_max_y - overlap_min_y)
                # assert 0 < overlap_area < area
                # ratio = overlap_area / area
                # ratio_total += ratio
                # image[y, x] += pix * ratio
                image[y, x] += pix * (overlap_area / area)

            x -= 1
            y -= 1

        # Top
        if y - 1 >= 0:
            y -= 1

            pix_min_x = xedges[x]
            pix_max_x = xedges[x + 1]
            pix_min_y = yedges[y + 1]
            pix_max_y = yedges[y]

            assert pix_max_x > pix_min_x and pix_max_y > pix_min_y

            overlap_min_x = max(min_x, pix_min_x)
            overlap_max_x = min(max_x, pix_max_x)
            overlap_min_y = max(min_y, pix_min_y)
            overlap_max_y = min(max_y, pix_max_y)

            if overlap_max_x > overlap_min_x and overlap_max_y > overlap_min_y:
                overlap_area = (overlap_max_x - overlap_min_x) * (overlap_max_y - overlap_min_y)
                # assert 0 < overlap_area < area
                # ratio = overlap_area / area
                # ratio_total += ratio
                # image[y, x] += pix * ratio
                image[y, x] += pix * (overlap_area / area)

            y += 1

        # Bottom
        if y + 1 < height:
            y += 1

            pix_min_x = xedges[x]
            pix_max_x = xedges[x + 1]
            pix_min_y = yedges[y + 1]
            pix_max_y = yedges[y]

            assert pix_max_x > pix_min_x and pix_max_y > pix_min_y

            overlap_min_x = max(min_x, pix_min_x)
            overlap_max_x = min(max_x, pix_max_x)
            overlap_min_y = max(min_y, pix_min_y)
            overlap_max_y = min(max_y, pix_max_y)

            if overlap_max_x > overlap_min_x and overlap_max_y > overlap_min_y:
                overlap_area = (overlap_max_x - overlap_min_x) * (overlap_max_y - overlap_min_y)
                # assert 0 < overlap_area < area
                # ratio = overlap_area / area
                # ratio_total += ratio
                # image[y, x] += pix * ratio
                image[y, x] += pix * (overlap_area / area)

            y -= 1

        # assert math.isclose(ratio_total, 1)

    return image
