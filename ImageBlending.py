import os
import imageio
import numpy as np
import scipy
from scipy.ndimage import filters
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


"""------------------------------------------------ magic numbers ---------------------------------------------------"""

MAX_INTENSITY = 255
GRAYSCALE_TYPE = 1
RGB_TYPE = 2
MIN_IMAGE_SIZE = 16

"""------------------------------------------- read_image from EX1 --------------------------------------------------"""


def read_image(filename, representation):
    """
    function which reads an image file and converts it into a given representation.
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image (1) or an RGB image (2).
    :return: image represented by a matrix of type np.float64 with intensities
    (either grayscale or RGB channel intensities) normalized to the range [0, 1].
    """
    if representation != GRAYSCALE_TYPE and representation != RGB_TYPE:
        exit(-1)

    image = imageio.imread(filename)
    image = np.array(image)

    # convert RGB to a grayscale image
    if representation == GRAYSCALE_TYPE:
        image = rgb2gray(image)

    # make sure that values are from type float64
    image = image.astype(np.float64)
    if image.max() > 1:
        image /= MAX_INTENSITY

    return image


""" --------------------------- load file --------------------------------------------------------- """


def relpath(filename):
    f = os.path.join(os.path.dirname(__file__), filename)
    return f


""" ------------------------------------------------------------------------------------------------ """


def create_gaussian_kernel(filter_size):
    """
    function to create row vector of shape (1, filter_size) used
    for the pyramid construction, using a consequent 1D convolutions of [1, 1] with itself
    in order to derive a row of the binomial coefficients.
    Then it normalized the vector.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter)
    :return: row vector of shape (1, filter_size) of binomial coefficients.
    """
    base_kernel = [1, 1]
    filter_vec = [1, 1]

    for i in range(1, filter_size - 1):
        filter_vec = np.convolve(filter_vec, base_kernel)

    sum_vec = sum(filter_vec)
    normalized_filter_vec = (filter_vec / sum_vec)
    return np.reshape(normalized_filter_vec, (1, filter_size))


def reduce(org_im, g_filter):
    """
     function that reduces image size by blurring and sampling every second pixel in every second line
    :param org_im: grayscale image from size N x M
    :param g_filter: row gaussian vector used for blurring the image
    :return: grayscale image from size N/2 x M/2
    """
    blur_row = scipy.ndimage.filters.convolve(org_im, g_filter)
    blur_col = scipy.ndimage.filters.convolve(blur_row, g_filter.T)
    reduce_image = blur_col[::2, ::2]
    return reduce_image


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    function to construct a Gaussian pyramid for an image
    :param im: grayscale image with values in range [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
    :return: pyr, filter_vec
             pyr -  a standard python array with maximum length of max_levels,
             where each element of the array is a grayscale image.
             filter_vec - row vector of shape (1, filter_size) used for the pyramid construction.
             (normalized).
    """
    filter_vec = create_gaussian_kernel(filter_size)
    pyr = [im]
    counter = 1
    while (counter < max_levels) and ((min((pyr[-1]).shape) // 2) >= MIN_IMAGE_SIZE):
        pyr.append(reduce(pyr[-1], filter_vec))
        counter += 1

    return pyr, filter_vec


def expand(org_im, g_filter):
    """
    function that expands image size by padding with zeros in every second pixel in every second line
    and blurring
    :param org_im: grayscale image from size N x M
    :param g_filter: row gaussian vector used for blurring the image
    :return: grayscale image from size (N * 2) x (M * 2)
    """
    expand_image = np.zeros((2 * org_im.shape[0], 2 * org_im.shape[1]))
    expand_image[::2, ::2] = org_im
    image_blur_row = scipy.ndimage.filters.convolve(expand_image, g_filter)
    image_blur_col = scipy.ndimage.filters.convolve(image_blur_row, g_filter.T)
    return image_blur_col


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    function to construct a Laplacian pyramid for an image
    :param im: grayscale image with values in range [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
    :return: pyr, filter_vec
             pyr -  a standard python array with maximum length of max_levels,
             where each element of the array is a grayscale image.
             filter_vec - row vector of shape (1, filter_size) used for the pyramid construction.
             (normalized).
    """
    g_pyr, g_filter = build_gaussian_pyramid(im, max_levels, filter_size)
    double_g_filter = 2 * g_filter
    pyr = []
    for i in range(len(g_pyr) - 1):
        pyr.append((g_pyr[i] - expand(g_pyr[i + 1], double_g_filter)))

    pyr.append(g_pyr[-1])
    return pyr, g_filter


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    function to reconstruct an image from its Laplacian Pyramid.
    :param lpyr: Laplacian pyramid.
    :param filter_vec: row vector of shape (1, filter_size) used for the pyramid construction.
    :param coeff: a python list, The list length is the same as the number of levels in the pyramid lpyr
    :return: grayscale image with values in range [0, 1]
    """
    for j in range(len(lpyr)):
        lpyr[j] *= coeff[j]

    double_filter_vec = filter_vec * 2
    revers_pyr = lpyr[::-1]

    for i in range(len(lpyr) - 1):
        revers_pyr[i] = expand(revers_pyr[i], double_filter_vec)
        revers_pyr[i + 1] = np.add(revers_pyr[i], revers_pyr[i + 1])

    return revers_pyr[-1]


def linear_scale(im):
    """
    function to stretch the values of a given image to range [0, 1]
    :param im: grayscale image
    :return: stretched image
    """
    min_val = im.min()
    scale_range = im.max() - min_val
    return (im - min_val) / scale_range


def render_pyramid(pyr, levels):
    """
    function to present a single black image in which the pyramid levels of the given pyramid are stacked horizontally.
    :param pyr: either a Gaussian or Laplacian pyramid
    :param levels: the number of levels to present in the result ≤ max_levels
    :return: res - single black image in which the pyramid levels of the given pyramid pyr are stacked horizontally
    """
    res = linear_scale(pyr[0])
    col = np.shape(res)[0]

    for i in range(1, levels):
        level = np.zeros((col, np.shape(pyr[i])[1]))
        level[:pyr[i].shape[0], :pyr[i].shape[1]] = linear_scale(pyr[i])
        res = np.concatenate((res, level), axis=1)

    return res


def display_pyramid(pyr, levels):
    """
    display of pyramids levels using "render_pyramid"
    :param pyr: either a Gaussian or Laplacian pyramid
    :param levels: the number of levels to present in the result ≤ max_levels
    :return: None
    """
    render_pyr = render_pyramid(pyr, levels)
    plt.imshow(render_pyr, cmap="gray")
    plt.show()


def builb_blend_laplacian(la, lb, mask):
    """
    performing pyramid blending on two images
    :param la: Laplacian pyramid of first image
    :param lb: Laplacian pyramid of second image
    :param mask: Gaussian pyramid of the mask
    :return: laplacian pyramid which is the blending of both images
    """
    l_pyr = []

    for i in range(len(mask)):
        l_pyr.append((np.multiply(mask[i], la[i]) + np.multiply((1 - mask[i]), lb[i])))

    return l_pyr


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    construct a Laplacian pyramid for blending on two images.
    :param im1: first input grayscale image to be blended.
    :param im2: second input grayscale image to be blended.
    :param mask: is a boolean (i.e. dtype == np.bool) mask containing True and False
           representing which parts of im1 and im2 should appear in the resulting im_blend
    :param max_levels: the max_levels parameter to use when generating the Gaussian and Laplacian pyramids
    :param filter_size_im: the size of the Gaussian filter (an odd scalar that represents a squared filter) which
           defining the filter used in the construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask: the size of the Gaussian filter(an odd scalar that represents a squared filter) which
           defining the filter used in the construction of the Gaussian pyramid of mask.
    :return: im_blend
    """
    la, f1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lb, f2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    mask_pyr, f3 = build_gaussian_pyramid(np.round(mask.astype("float64")), max_levels, filter_size_mask)
    lc = builb_blend_laplacian(la, lb, mask_pyr)
    image = laplacian_to_image(lc, f1, [1]*max_levels)
    return np.clip(image, 0, 1)


def blending_example1():
    """
    performing pyramid blending on two images I found nice.
    :return: im1 - first grayscale image with values in range [0, 1]
             im2 - second grayscale image with values in range [0, 1]
             mask - binary mask
             im_blend - blended image
    """
    im1 = read_image(relpath("externals/volcan.png"), 2)
    im2 = read_image(relpath("externals/popcorn1.png"), 2)
    mask = read_image(relpath("externals/popcornMask.png"), 1)

    b_image_r = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask.astype("bool"), 2, 3, 11)
    b_image_g = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask.astype("bool"), 2, 3, 11)
    b_image_b = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask.astype("bool"), 2, 3, 11)

    b_image = np.zeros(np.shape(im1))
    b_image[:, :, 0] = b_image_r
    b_image[:, :, 1] = b_image_g
    b_image[:, :, 2] = b_image_b

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(im1)
    axs[0, 0].set_title('im1')
    axs[0, 1].imshow(im2)
    axs[0, 1].set_title('im2')
    axs[1, 0].imshow(mask, cmap="gray")
    axs[1, 0].set_title('mask')
    axs[1, 1].imshow(b_image)
    axs[1, 1].set_title('b_image')
    plt.show()

    return im1, im2, np.round(mask).astype("bool"), b_image


def blending_example2():
    """
    performing pyramid blending on two images I found nice.
    :return: im1 - first grayscale image with values in range [0, 1]
             im2 - second grayscale image with values in range [0, 1]
             mask - binary mask
             im_blend - blended image
    """
    im1 = read_image(relpath("externals/pool_project.png"), 2)
    im2 = read_image(relpath("externals/donut.png"), 2)
    mask = read_image(relpath("externals/poolMask.png"), 1)

    b_image_r = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask.astype("bool"), 3, 5, 5)
    b_image_g = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask.astype("bool"), 3, 5, 5)
    b_image_b = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask.astype("bool"), 3, 5, 5)

    b_image = np.zeros(np.shape(im1))
    b_image[:, :, 0] = b_image_r
    b_image[:, :, 1] = b_image_g
    b_image[:, :, 2] = b_image_b

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(im1)
    axs[0, 0].set_title('im1')
    axs[0, 1].imshow(im2)
    axs[0, 1].set_title('im2')
    axs[1, 0].imshow(mask, cmap="gray")
    axs[1, 0].set_title('mask')
    axs[1, 1].imshow(b_image)
    axs[1, 1].set_title('b_image')
    plt.show()

    return im1, im2, np.round(mask).astype("bool"), b_image
