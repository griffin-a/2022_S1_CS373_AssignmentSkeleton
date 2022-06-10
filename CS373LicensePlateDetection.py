import math
import sys
from pathlib import Path

from matplotlib import pyplot
from matplotlib.patches import Rectangle
from PIL import Image

# import our basic, light-weight png reader library
import imageIO.png
import easyocr


class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):
    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue=0):
    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    f_low = f_high = None
    res = createInitializedGreyscalePixelArray(image_width, image_height)

    for row in pixel_array:
        for col in row:
            if f_low is None: f_low = col
            if f_high is None: f_high = col

            f_low = min(f_low, col)
            f_high = max(f_high, col)

    g_min, g_max = 0, 255
    divisor = (g_max / (f_high - f_low)) if f_high - f_low > 0 else round(g_max)
    a = divisor
    b = (g_min - f_low) * divisor

    for i in range(len(pixel_array)):
        for j in range(len(pixel_array[i])):
            if pixel_array[i][j] > f_high:
                res[i][j] = g_max
            elif f_low <= pixel_array[i][j] <= f_high:
                temp = round((a * pixel_array[i][j]) + b)
                res[i][j] = temp if 0 <= temp <= 255 else pixel_array[i][j]
            elif pixel_array[i][j] < f_low:
                res[i][j] = g_min

    return res


def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    res = [[0 for x in range(image_width)] for y in range(image_height)]

    for i in range(1, image_height - 2):
        for j in range(1, image_width - 2):
            box = []

            for eta in [-2, -1, 0, 1, 2]:
                for xi in [-2, -1, 0, 1, 2]:
                    box.append(pixel_array[i + eta][j + xi])

            mean = sum(box) / 25
            points_sum = 0

            for val in box:
                points_sum += (val - mean) ** 2

            variance = points_sum / 25
            std_dev = math.sqrt(variance)

            res[i][j] = round(std_dev, 3)

    return res


def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    res = createInitializedGreyscalePixelArray(image_width, image_height)

    for row in range(len(pixel_array)):
        for col in range(len(pixel_array[row])):
            if pixel_array[row][col] < threshold_value:
                res[row][col] = 0
            else:
                res[row][col] = 255

    return res


def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    res = [[0 for x in range(image_width)] for y in range(image_height)]
    for i in range(image_height):
        for j in range(image_width):
            # if pixel_array[i][j] == 255:
            #     pixel_array[i][j] = 1
            if i + 1 > image_height - 1 or i - 1 < 0 or j + 1 > image_width - 1 or j - 1 < 0:
                res[i][j] = 0
            else:
                if (pixel_array[i - 1][j - 1] != 0 and pixel_array[i - 1][j] != 0 and pixel_array[i - 1][j + 1] != 0 and
                        pixel_array[i][j - 1] != 0 and pixel_array[i][j] != 0 and pixel_array[i][j + 1] != 0 and
                        pixel_array[i + 1][j - 1] != 0 and pixel_array[i + 1][j] != 0 and pixel_array[i + 1][
                            j + 1] != 0):
                    res[i][j] = 1
                else:
                    res[i][j] = 0
    return res


def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    res = [[0 for x in range(image_width)] for y in range(image_height)]
    for i in range(image_height):
        for j in range(image_width):

            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    if 0 <= i + x < image_height - 1 and 0 <= j + y < image_width - 1:
                        if pixel_array[i + x][j + y] != 0:
                            res[i][j] = 1
                        # else:
                        #     res[i][j] = 0

    return res


def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    res = [[0 for x in range(image_width)] for y in range(image_height)]
    visited = {(x, y): False for x in range(image_height) for y in range(image_width)}
    d = {}
    q = Queue()

    curr_component = 1

    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] != 0 and not visited[(i, j)]:
                q.enqueue((i, j))
                d[curr_component] = 0

                while not q.isEmpty():
                    u = q.dequeue()

                    x, y = u[0], u[1]
                    visited[(x, y)] = True

                    res[x][y] = curr_component
                    d[curr_component] += 1

                    if x - 1 >= 0:
                        if pixel_array[x - 1][y] != 0 and not visited[(x - 1, y)]:
                            q.enqueue((x - 1, y))
                            visited[(x - 1, y)] = True

                    if x + 1 < image_height:
                        if pixel_array[x + 1][y] != 0 and not visited[(x + 1, y)]:
                            q.enqueue((x + 1, y))
                            visited[(x + 1, y)] = True

                    if y - 1 >= 0:
                        if pixel_array[x][y - 1] != 0 and not visited[(x, y - 1)]:
                            q.enqueue((x, y - 1))
                            visited[(x, y - 1)] = True

                    if y + 1 < image_width:
                        if pixel_array[x][y + 1] != 0 and not visited[(x, y + 1)]:
                            q.enqueue((x, y + 1))
                            visited[(x, y + 1)] = True

                curr_component += 1

    return res, d


# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!
def main():
    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate3.png"

    if command_line_arguments:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(input_filename.replace(".png", "_output.png"))
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Greyscale')
    axs1[0, 0].imshow(px_array_r, cmap='gray')

    # STUDENT IMPLEMENTATION here
    px_array = scaleTo0And255AndQuantize(px_array_r, image_width, image_height)
    px_array = computeStandardDeviationImage5x5(px_array, image_width, image_height)
    px_array = scaleTo0And255AndQuantize(px_array, image_width, image_height)

    axs1[0, 1].set_title('Contrast Stretching')
    axs1[0, 1].imshow(px_array, cmap='gray')
    # 140 5 6 (4/6)
    # 140 3 3 (2 and 3 are close; 4 only 2/3 match, 1, 5, 6 match)
    # 150 4 4 (4/6) (2 and 3 are somewhat close)
    # 150 3 5, 5x5 dilation (4/6)
    px_array = computeThresholdGE(px_array, 150, image_width, image_height)

    # print(px_array)
    for _ in range(3):
        px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)

    for _ in range(3):
        px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)

    # (px_array)
    # px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)

    axs1[1, 0].set_title('Dilation and erosion')
    axs1[1, 0].imshow(px_array, cmap='gray')

    res, d = computeConnectedComponentLabeling(px_array, image_width, image_height)
    print(d)

    # Find the largest connected component
    largest_key = max(d, key=d.get)
    print(largest_key)

    min_pixel, max_pixel = None, None

    reversed_dict = sorted(d.items(), key=lambda x: x[1], reverse=True)
    reversed_keys = [x[0] for x in reversed_dict]
    print(reversed_keys)

    largest_key_ar = None

    for key in reversed_keys:
        x_list, y_list = [], []
        for i in range(len(res)):
            for j in range(len(res[i])):
                if res[i][j] == key:
                    # if not min_pixel:
                    #     min_pixel = [i, j]
                    # elif i > max_pixel[0]:
                    #     max_pixel[0] = i
                    # elif j > max_pixel[1]:
                    #     max_pixel[1] = j
                    # max_pixel[0] = max(max_pixel[0], i)
                    # max_pixel[1] = max(max_pixel[1], j)
                    x_list.append(i)
                    y_list.append(j)

        min_pixel, max_pixel = [min(x_list), min(y_list)], [max(x_list), max(y_list)]
        bbox_width = max_pixel[1] - min_pixel[1]
        bbox_height = max_pixel[0] - min_pixel[0]
        aspect_ratio = bbox_width / bbox_height
        # (470, 225) and (590, 270)
        if 1.5 <= aspect_ratio <= 5:
            print("Aspect ratio hit")
            largest_key_ar = key
            break

    print("largest key with AR", largest_key_ar)
    print("min pixel", min_pixel)
    print("max pixel", max_pixel)

    # with open('./output.txt', 'w') as f:
    #     for row in res:
    #         f.write(str(row))

    # compute a dummy bounding box centered in the middle of the input image, and with a size of half of width and
    # height
    center_x = image_width / 2.0
    center_y = image_height / 2.0
    bbox_min_x = center_x - image_width / 4.0
    bbox_max_x = center_x + image_width / 4.0
    bbox_min_y = center_y - image_width / 4.0
    bbox_max_y = center_y + image_width / 4.0

    # Binary `BW image of only the largest connected component: testing purposes only
    # res_largest_component = createInitializedGreyscalePixelArray(image_width, image_height)
    #
    # for i in range(len(res)):
    #     for j in range(len(res[i])):
    #         if res[i][j] == largest_key_ar:
    #             res_largest_component[i][j] = 1
    # print(res_largest_component)

    image = Image.open(input_filename)
    cropped_image = image.crop((min_pixel[1], min_pixel[0], max_pixel[1], max_pixel[0]))
    new_name = "{}-cropped.png".format(input_filename.replace(".png", ""))
    cropped_image.save(new_name)
    # reader = easyocr.Reader(['en'])
    # results = reader.readtext(f"./{input_filename}")
    # print(results)

    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image')
    axs1[1, 1].imshow(px_array_r, cmap='gray')
    rect = Rectangle((min_pixel[1], min_pixel[0]), max_pixel[1] - min_pixel[1], max_pixel[0] - min_pixel[0],
                     linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()
