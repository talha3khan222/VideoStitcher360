from Utils.FeatureMatcher import FeatureMatcher
import numpy as np
import cv2


def transform_with_homography(h_mat, points_array):
    """Function to transform a set of points using the given homography matrix.
        Points are normalized after transformation with the last column which represents the scale

    Args:
        h_mat (numpy array): of shape (3, 3) representing the homography matrix
        points_array (numpy array): of shape (n, 2) represting n set of x, y pixel coordinates that are
            to be transformed
    """
    # add column of ones so that matrix multiplication with homography matrix is possible
    ones_col = np.ones((points_array.shape[0], 1))
    points_array = np.concatenate((points_array, ones_col), axis=1)
    transformed_points = np.matmul(h_mat, points_array.T)
    epsilon = 1e-7  # very small value to use it during normalization to avoid division by zero
    transformed_points = transformed_points / (transformed_points[2, :].reshape(1, -1) + epsilon)
    transformed_points = transformed_points[0:2, :].T
    return transformed_points


def compute_outliers(h_mat, points_img_a, points_img_b, threshold=3):
    '''Function to compute the error in the Homography matrix using the matching points in
        image A and image B

    Args:
        h_mat (numpy array): of shape (3, 3) representing the homography that transforms points in image B to points in image A
        points_img_a (numpy array): of shape (n, 2) representing pixel coordinate points (u, v) in image A
        points_img_b (numpy array): of shape (n, 2) representing pixel coordinates (x, y) in image B
        theshold (int): a number that represents the allowable euclidean distance (in pixels) between the transformed pixel coordinate from
            the image B to the matched pixel coordinate in image A, to be conisdered outliers

    Returns:
        error: a scalar float representing the error in the Homography matrix
    '''
    num_points = points_img_a.shape[0]
    outliers_count = 0

    # transform the match point in image B to image A using the homography
    points_img_b_hat = transform_with_homography(h_mat, points_img_b)

    # let x, y be coordinate representation of points in image A
    # let x_hat, y_hat be the coordinate representation of transformed points of image B with respect to image A
    x = points_img_a[:, 0]
    y = points_img_a[:, 1]
    x_hat = points_img_b_hat[:, 0]
    y_hat = points_img_b_hat[:, 1]
    euclid_dis = np.sqrt(np.power((x_hat - x), 2) + np.power((y_hat - y), 2)).reshape(-1)
    for dis in euclid_dis:
        if dis > threshold:
            outliers_count += 1
    return outliers_count


def calculate_homography(points_img_a, points_img_b):
    '''Function to calculate the homography matrix from point corresspondences using Direct Linear Transformation
        The resultant homography transforms points in image B into points in image A
        Homography H = [h1 h2 h3;
                        h4 h5 h6;
                        h7 h8 h9]
        u, v ---> point in image A
        x, y ---> matched point in image B then,
        with n point correspondences the DLT equation is:
            A.h = 0
        where A = [-x1 -y1 -1 0 0 0 u1*x1 u1*y1 u1;
                   0 0 0 -x1 -y1 -1 v1*x1 v1*y1 v1;
                   ...............................;
                   ...............................;
                   -xn -yn -1 0 0 0 un*xn un*yn un;
                   0 0 0 -xn -yn -1 vn*xn vn*yn vn]
        This equation is then solved using SVD
        (At least 4 point correspondences are required to determine 8 unkwown parameters of homography matrix)
    Args:
        points_img_a (numpy array): of shape (n, 2) representing pixel coordinate points (u, v) in image A
        points_img_b (numpy array): of shape (n, 2) representing pixel coordinates (x, y) in image B

    Returns:
        h_mat: A (3, 3) numpy array of estimated homography
    '''
    # concatenate the two numpy points array to get 4 columns (u, v, x, y)
    points_a_and_b = np.concatenate((points_img_a, points_img_b), axis=1)
    A = []
    # fill the A matrix by looping through each row of points_a_and_b containing u, v, x, y
    # each row in the points_ab would fill two rows in the A matrix
    for u, v, x, y in points_a_and_b:
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

    A = np.array(A)
    _, _, v_t = np.linalg.svd(A)

    # soltion is the last column of v which means the last row of its transpose v_t
    h_mat = v_t[-1, :].reshape(3, 3)
    return h_mat


class HomographyCalculator:
    def __init__(self, conf_threshold):
        self._confidence_threshold = conf_threshold

    def compute_homography_ransac(self, matches1, matches2):
        num_all_matches = matches1.shape[0]
        # RANSAC parameters
        SAMPLE_SIZE = 5  # number of point correspondances for estimation of Homgraphy
        SUCCESS_PROB = 0.995  # required probabilty of finding H with all samples being inliners
        min_iterations = int(np.log(1.0 - SUCCESS_PROB) / np.log(1 - 0.5 ** SAMPLE_SIZE))

        # Let the initial error be large i.e consider all matched points as outliers
        lowest_outliers_count = num_all_matches
        best_h_mat = None
        best_i = 0  # just to know in which iteration the best h_mat was found

        for i in range(min_iterations):
            rand_ind = np.random.permutation(range(num_all_matches))[:SAMPLE_SIZE]
            h_mat = calculate_homography(matches1[rand_ind], matches2[rand_ind])
            outliers_count = compute_outliers(h_mat, matches1, matches2)
            if outliers_count < lowest_outliers_count:
                best_h_mat = h_mat
                lowest_outliers_count = outliers_count
                best_i = i
        best_confidence_obtained = int(100 - (100 * lowest_outliers_count / num_all_matches))
        if best_confidence_obtained < self._confidence_threshold:
            return None

        return best_h_mat


def get_crop_points_vert(img_a_w, transfmd_corners_img_b):
    """Function to find the pixel corners in the vertically stitched images to crop and remove the
        black space around.

    Args:
        img_a_h (int): the width of the pivot image that is image A
        transfmd_corners_img_b (numpy array): of shape (n, 2) representing the transformed corners of image B
            The corners need to be in the following sequence:
            corners = [top_left_x, top_left_y;
                   top_right_x, top_right_y;
                   bottom_right_x, bottom_right_y;
                   bottom_left_x, bottom_left_y]
    Returns:
        x_start (int): the x pixel-cordinate to start the crop on the stitched image
        y_start (int): the x pixel-cordinate to start the crop on the stitched image
        x_end (int): the x pixel-cordinate to end the crop on the stitched image
        y_end (int): the y pixel-cordinate to end the crop on the stitched image
    """
    # the four transformed corners of image B
    top_lft_x_hat, top_lft_y_hat = transfmd_corners_img_b[0, :]
    top_rht_x_hat, top_rht_y_hat = transfmd_corners_img_b[1, :]
    btm_rht_x_hat, btm_rht_y_hat = transfmd_corners_img_b[2, :]
    btm_lft_x_hat, btm_lft_y_hat = transfmd_corners_img_b[3, :]

    # initialize the crop points
    # since image A (on the top) is used as pivot, y_start will always be zero
    x_start, y_start, x_end, y_end = (None, 0, None, None)

    if (top_lft_x_hat > 0) and (top_lft_x_hat > btm_lft_x_hat):
        x_start = top_lft_x_hat
    elif (btm_lft_x_hat > 0) and (btm_lft_x_hat > top_lft_x_hat):
        x_start = btm_lft_x_hat
    else:
        x_start = 0

    if (top_rht_x_hat < img_a_w - 1) and (top_rht_x_hat < btm_rht_x_hat):
        x_end = top_rht_x_hat
    elif (btm_rht_x_hat < img_a_w - 1) and (btm_rht_x_hat < top_rht_x_hat):
        x_end = btm_rht_x_hat
    else:
        x_end = img_a_w - 1

    if btm_lft_y_hat < btm_rht_y_hat:
        y_end = btm_lft_y_hat
    else:
        y_end = btm_rht_y_hat

    return int(x_start), int(y_start), int(x_end), int(y_end)


def get_crop_points_horz(img_a_h, transfmd_corners_img_b):
    """Function to find the pixel corners in the horizontally stitched images to crop and remove the
        black space around.

    Args:
        img_a_h (int): the height of the pivot image that is image A
        transfmd_corners_img_b (numpy array): of shape (n, 2) representing the transformed corners of image B
            The corners need to be in the following sequence:
            corners = [top_left_x, top_left_y;
                   top_right_x, top_right_y;
                   bottom_right_x, bottom_right_y;
                   bottom_left_x, bottom_left_y]
    Returns:
        x_start (int): the x pixel-cordinate to start the crop on the stitched image
        y_start (int): the x pixel-cordinate to start the crop on the stitched image
        x_end (int): the x pixel-cordinate to end the crop on the stitched image
        y_end (int): the y pixel-cordinate to end the crop on the stitched image
    """
    # the four transformed corners of image B
    top_lft_x_hat, top_lft_y_hat = transfmd_corners_img_b[0, :]
    top_rht_x_hat, top_rht_y_hat = transfmd_corners_img_b[1, :]
    btm_rht_x_hat, btm_rht_y_hat = transfmd_corners_img_b[2, :]
    btm_lft_x_hat, btm_lft_y_hat = transfmd_corners_img_b[3, :]

    # initialize the crop points
    # since image A (on the left side) is used as pivot, x_start will always be zero
    x_start, y_start, x_end, y_end = (0, None, None, None)

    if (top_lft_y_hat > 0) and (top_lft_y_hat > top_rht_y_hat):
        y_start = top_lft_y_hat
    elif (top_rht_y_hat > 0) and (top_rht_y_hat > top_lft_y_hat):
        y_start = top_rht_y_hat
    else:
        y_start = 0

    if (btm_lft_y_hat < img_a_h - 1) and (btm_lft_y_hat < btm_rht_y_hat):
        y_end = btm_lft_y_hat
    elif (btm_rht_y_hat < img_a_h - 1) and (btm_rht_y_hat < btm_lft_y_hat):
        y_end = btm_rht_y_hat
    else:
        y_end = img_a_h - 1

    if top_rht_x_hat < btm_rht_x_hat:
        x_end = top_rht_x_hat
    else:
        x_end = btm_rht_x_hat

    return int(x_start), int(y_start), int(x_end), int(y_end)


def get_corners_as_array(img_height, img_width):
    """Function to extract the corner points of an image from its width and height and arrange it in the form
        of a numpy array.

        The 4 corners are arranged as follows:
        corners = [top_left_x, top_left_y;
                   top_right_x, top_right_y;
                   bottom_right_x, bottom_right_y;
                   bottom_left_x, bottom_left_y]

    Args:
        img_height (str): height of the image
        img_width (str): width of the image

    Returns:
        corner_points_array (numpy array): of shape (4,2) representing for corners with x,y pixel coordinates
    """
    corners_array = np.array([[0, 0],
                              [img_width - 1, 0],
                              [img_width - 1, img_height - 1],
                              [0, img_height - 1]])
    return corners_array


class ImagesJoiner:
    def __init__(self):
        pass

    def get_crop_points(self, h_mat, img_a, img_b, stitch_direc):
        """Function to find the pixel corners to crop the stitched image such that the black space
            in the stitched image is removed.
            The black space could be because either image B is not of the same dimensions as image A
            or image B is skewed after homographic transformation.
            Example:
                      (Horizontal stitching)
                    ____________                     _________________
                    |           |                    |                |
                    |           |__________          |                |
                    |           |         /          |       A        |
                    |     A     |   B    /           |________________|
                    |           |       /                |          |
                    |           |______/                 |    B     |
                    |___________|                        |          |
                                                         |__________|  <-imagine slant bottom edge

            This function returns the corner points to obtain the maximum area inside A and B combined and making
            sure the edges are straight (i.e horizontal and veritcal).

        Args:
            h_mat (numpy array): of shape (3, 3) representing the homography from image B to image A
            img_a (numpy array): of shape (h, w, c) representing image A
            img_b (numpy array): of shape (h, w, c) representing image B
            stitch_direc (int): 0 when stitching vertically and 1 when stitching horizontally

        Returns:
            x_start (int): the x pixel-cordinate to start the crop on the stitched image
            y_start (int): the x pixel-cordinate to start the crop on the stitched image
            x_end (int): the x pixel-cordinate to end the crop on the stitched image
            y_end (int): the y pixel-cordinate to end the crop on the stitched image
        """
        img_a_h, img_a_w, _ = img_a.shape
        img_b_h, img_b_w, _ = img_b.shape

        orig_corners_img_b = get_corners_as_array(img_b_h, img_b_w)

        transfmd_corners_img_b = transform_with_homography(h_mat, orig_corners_img_b)

        if stitch_direc == 1:
            x_start, y_start, x_end, y_end = get_crop_points_horz(img_a_w, transfmd_corners_img_b)
        # initialize the crop points
        x_start = None
        x_end = None
        y_start = None
        y_end = None

        if stitch_direc == 1:  # 1 is horizontal
            x_start, y_start, x_end, y_end = get_crop_points_horz(img_a_h, transfmd_corners_img_b)
        else:  # when stitching images in the vertical direction
            x_start, y_start, x_end, y_end = get_crop_points_vert(img_a_w, transfmd_corners_img_b)
        return x_start, y_start, x_end, y_end


class StitchImages:
    def __init__(self):
        self.features_matcher = FeatureMatcher()
        self._homography_calculator = HomographyCalculator(65)
        self._images_joiner = ImagesJoiner()
        self._homography_matrix = None
        self._best_matches_count = [0, 0]

    def stitch(self, image1, image2, stitch_direction=1):

        matches1, matches2 = self.features_matcher.match_images(image1, image2)

        if matches1 is None or matches2 is None:
            return None

        matching_average1 = np.average(matches1, 0)
        matching_average2 = np.average(matches2, 0)

        if matching_average1[stitch_direction] >= image1.shape[stitch_direction] / 2:
            img_a = image1
            img_b = image2
            matches_a = matches1
            matches_b = matches2
        else:
            img_a = image2
            img_b = image1
            matches_a = matches2
            matches_b = matches1

        for mat in matches_a:
            mat = [round(point) for point in mat]
            image1 = cv2.circle(img_a, mat, 2, (0, 255, 0), -1)
        for mat in matches_b:
            mat = [round(point) for point in mat]
            image2 = cv2.circle(img_b, mat, 2, (0, 255, 255), -1)

        cv2.imshow('image1', image1)
        cv2.imshow('image2', image2)

        if self._homography_matrix is None or (self._homography_matrix is not None and
                                               (len(matches_a) > self._best_matches_count[0] or
                                                len(matches_b) > self._best_matches_count[1])):
            self._homography_matrix = self._homography_calculator.compute_homography_ransac(matches_a, matches_b)
            self._best_matches_count[0] = len(matches_a)
            self._best_matches_count[1] = len(matches_b)

        # self._homography_matrix = self._homography_calculator.compute_homography_ransac(matches_a, matches_b)

        # Warp source image to destination based on homography
        #im_out = cv2.warpPerspective(img_a, self._homography_matrix, (img_b.shape[1], img_b.shape[0]))

        #cv2.imshow('imout', im_out)

        overlapping_area_a = img_a[:, np.where(matches_a[:, 1] == np.min(matches_a[:, 1])):]

        if self._homography_matrix is None:
            print("Not enough matching points")
            return None

        if stitch_direction == 0:
            canvas = cv2.warpPerspective(img_b, self._homography_matrix,
                                         (img_a.shape[1], img_a.shape[0] + img_b.shape[0]))
            canvas[0:img_a.shape[0], :, :] = img_a[:, :, :]
            x_start, y_start, x_end, y_end = self._images_joiner.get_crop_points(self._homography_matrix,
                                                                                 img_a, img_b, 0)
        else:
            canvas = cv2.warpPerspective(img_b, self._homography_matrix,
                                         (img_a.shape[1] + img_b.shape[1], img_a.shape[0]))
            canvas[:, 0:img_a.shape[1], :] = img_a[:, :, :]
            x_start, y_start, x_end, y_end = self._images_joiner.get_crop_points(self._homography_matrix,
                                                                                 img_a, img_b, 1)

        stitched_img = canvas[y_start:y_end, x_start:x_end, :]
        cv2.imshow('canvas', canvas)
        return stitched_img



