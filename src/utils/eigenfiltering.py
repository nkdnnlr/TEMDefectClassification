import os
import time

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tqdm

# import src.utils.helpers as helpers
import utils.helpers as helpers
# import helpers as helpers



def get_fft_period(x):
    """
    Return most occuring frequency in x or y direction
    :param x:
    :return:
    """
    N = x[0].size
    f = np.linspace(1, N//2-1, N//2-1)
    sum_spec = np.zeros(N)
    for row in x:
        fft = np.fft.fft(row)
        sum_spec += np.abs(fft)#[1:N//2]
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency [Hz]")
    plt.plot(sum_spec[:N//2], '.')
    plt.show()
    sorted_indices = np.argsort(-sum_spec[:N//2])
    return sorted_indices[1]


def get_filter_fixed(patch, size):
    """
    Get filter with fixed size
    :param patch:
    :param size:
    :return:
    """
    samples = []
    for i in range(patch.shape[0] - size + 1):
        for j in range(patch.shape[1] - size + 1):
            sample = []
            for h in range(size):
                for k in range(size):
                    sample.append(patch[i + h, j + k])
            samples.append(sample)
    return samples


def get_filter_sparse(patch, period_y, period_x):
    """
    Get sparse filter with 9 entries according to periodicity
    :param patch: Patch to create filter from
    :param period_x: Periodicty in x
    :param period_y: Periodicty in y
    :return:
    """
    samples = []
    for i in range(patch.shape[0] - period_y):
        for j in range(patch.shape[1] - period_x):
            sample = np.array(
                [
                    patch[i + 0, j + 0],
                    patch[i + 0, j + int(0.5 * period_x)],
                    patch[i + 0, j + period_x - 1],
                    patch[i + int(0.5 * period_y), j + 0],
                    patch[i + int(0.5 * period_y), j + int(0.5 * period_x)],
                    patch[i + int(0.5 * period_y), j + period_x - 1],
                    patch[i + period_y - 1, j + 0],
                    patch[i + period_y - 1, j + int(0.5 * period_x)],
                    patch[i + period_y - 1, j + period_x - 1],
                ]
            )
            samples.append(sample)
    return samples


def get_eigenvectors(samples):
    """
    Get list of eigenvectors and list of eigenvalues, both ascending by eigenvalue
    :param samples: Sample vector
    :return:
    """
    # Covariance matrix
    covariance = np.cov(samples.T)

    # Get Eigenvectors&Eigenvalues
    w, v = np.linalg.eig(covariance)
    w_indices = w.argsort()
    w_sorted = w[w_indices[::-1]]
    v_sorted = v[w_indices[::-1]]
    return v_sorted, w_sorted


def build_dense_filter(v_sorted):
    """
    Build a dense filter from list of eigenvectors.
    :param v_sorted: List of eigenvector array
    :return:
    """
    filters = []
    for eigvec in v_sorted:
        s = eigvec.shape[0]
        fs = int(np.sqrt(s))
        filter = np.zeros([fs, fs])
        for i in range(s):
            filter[i // fs, i % fs] = eigvec[i]
        filters.append(filter)
    filters = np.array(filters)
    return filters

def build_sparse_filter(v_sorted, period_y, period_x):
    """
    Build a sparse filter from list of eigenvectors. Usually not used, see below.
    :param v_sorted: List of eigenvector array
    :param period_y: Periodicity in y
    :param period_x: Periodicity in x
    :return:
    """
    filters = []
    for eigvec in v_sorted:
        filter = np.zeros([period_y, period_x])
        filter[0, 0] = eigvec[0]
        filter[0, int(0.5 * period_x)] = eigvec[1]
        filter[0, period_x - 1] = eigvec[2]
        filter[int(0.5 * period_y), 0] = eigvec[3]
        filter[int(0.5 * period_y), int(0.5 * period_x)] = eigvec[4]
        filter[int(0.5 * period_y), period_x - 1] = eigvec[5]
        filter[period_y - 1, 0] = eigvec[6]
        filter[period_y - 1, int(0.5 * period_x)] = eigvec[7]
        filter[period_y - 1, period_x - 1] = eigvec[8]
        filters.append(filter)
    filters = np.array(filters)
    return filters


def convolute_filter(patch, filter):
    """
    Convolve image patch with filter
    :param patch: Image patch
    :param filter: Filter
    :return:
    """
    sample = sp.ndimage.filters.convolve(patch, filter, mode="mirror")  # , cval=1.0)
    # sample = sp.ndimage.filters.convolve(patch, filter, mode="wrap")  # , cval=1.0)
    # sample = sp.ndimage.filters.convolve(patch, filter, mode="constant", cval=2550.)

    return sample


def get_energy(image, blur=25):
    """
    Create image with more homogeneous intensities by calculating the "energy" of the original image.
    Convolve the squared image with a uniform blurring filter.
    :param image: Original image
    :param blur: Size of blurring filter
    :return:
    """
    energy = sp.ndimage.filters.convolve(np.square(image), np.ones([blur, blur]), mode="constant")  # , cval=1.0)
    # energy = np.square(image)
    return energy


def mahalanobis_matrix(sample, reference):
    """
    Calculate the Mahalanobis distance of two images with multiple channels, e.g. the PCA energies
    :param sample: Sample image
    :param reference: Reference image
    :return:
    """
    assert sample.shape == reference.shape, "Not the same size!"
    p, m, n = reference.shape
    reference_1d = np.reshape(reference, (p, m * n)).T
    sample_1d = np.reshape(sample, (p, m * n)).T
    reference_cov = np.cov(reference_1d, rowvar=False)
    reference_cov_inv = np.linalg.inv(reference_cov)

    mh = np.zeros([m * n])
    for pixel in range(m * n):
        mh_px = sp.spatial.distance.mahalanobis(
            sample_1d[pixel, :], reference_1d[pixel, :], reference_cov_inv
        )
        mh[pixel] = mh_px

    mh = np.reshape(mh, (m, n))
    return mh

def handle_edges(image, edgesize, patchsize):
    e = edgesize
    p = patchsize

    lenx, leny = image.shape

    for x in range(lenx):
        for y in range(leny):
            for n in range(24):
                if (x >= e+(1/2+1/4+n/2)*p) and (x < e+(1/2+2/4+n/2)*p):
                    image[x,y] *= 2
            # for n in range(12):
                if (y >= e+(1/2+1/4+n/2)*p) and (y < e+(1/2+2/4+n/2)*p):
                    image[x,y] *= 2
                    
            if (x < e+(1/2)*p) or (y < e+(1/2)*p) or (x > lenx-e-(1/4)*p) or (y > leny-e-(1/4)*p):
                image[x,y] *= 2
                
            
            if (x < e+(1/2)*p) and (y < e+(1/2)*p):
                image[x,y] *= 2
                
            if (x < e+(1/2)*p) and (y > leny-e-(1/4)*p):
                image[x,y] *= 2
                
            if (x > lenx-e-(1/4)*p) and (y < e+(1/2)*p):
                image[x,y] *= 2
            
            if (x > lenx-e-(1/4)*p) and (y > leny-e-(1/4)*p):
                image[x,y] *= 2    
                
            
            
    return image
    

def eigenfiltering(image_def, patch_good, filter_fixed=True, filter_size=7, blurring=16, output_path=None):
    """
    Applies the eigenfiltering algorithm to an image, given a good patch
    :param image_def: Image to be filtered
    :param patch_good: Good patch to design the filter from
    :param filter_fixed: Boolean. If False, choosing sparse filter based on periodicity (calculating periodicity does not work well)
    :param filter_size: Filter size (square) TODO: check again optimum
    :param blurring: Blur factor used in calculating the energies TODO: check again optimum
    :return:
    """
    # blurring = 2
    target_size = patch_good.shape[0]
    edge = 16 // 2 + 8
    step = target_size // 2 #- 2 * edge

    starttime = time.time()

    if filter_fixed:
        # Get filter fixed sized
        samples = get_filter_fixed(patch_good, filter_size)
        samples = np.array(samples)
        # print(samples.shape)

        # Get Eigenvectors
        v_sorted, w_sorted = get_eigenvectors(samples)

        # Get dense filter
        filters = build_dense_filter(v_sorted)

    else:
        # Get period from fft
        period_x = get_fft_period(patch_good)
        period_y = get_fft_period(patch_good.T)

        print("period x:", period_x, "period y:", period_y)

        # Get filter custom sized
        samples = get_filter_sparse(patch_good, period_y=period_y, period_x=period_x)
        samples = np.array(samples)
        # print(samples.shape)

        # Get Eigenvectors
        v_sorted, w_sorted = get_eigenvectors(samples)

        # Get Sparse filters
        filters = build_sparse_filter(v_sorted, period_y, period_x)

    # Save filters
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if output_path is not None:
        for idx, filter in enumerate(filters):
            plt.imshow(filter, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(output_path, str(idx) + '.svg'), bbox_inches='tight')#,transparent=True,
            # pad_inches=0)
            plt.close()
            # plt.show()
        # exit()

    # Convolute Filter with good images
    good_filtered_all = []
    good_energies = []
    i = 0
    for filter in filters:
        good_filtered = convolute_filter(patch_good, filter)
        good_filtered_all.append(good_filtered)
        energy = get_energy(good_filtered, blur=blurring)
        good_energies.append(energy)
        i += 1
    good_energies = np.array(good_energies)
    
    # fig, ax = plt.subplots(ncols=4)
    
    # print(patch_good)
    # print("Ho")
    # print(filter)
    # print("Ho")
    # # print(good_filtered)
    # # orint("Ho")
    # exit()    
    # ax[0].imshow(patch_good)
    # ax[1].imshow(filter)
    # im2 = ax[2].imshow(good_filtered)
    # cbar0 = fig.colorbar(im2, ax=ax[2])
    # ax[3].imshow(energy)
    # plt.show()
    # exit()
    
    patches_def = helpers.get_patches(image_def, target_size=target_size, step=step)
    n_patches = len(patches_def)

    mhs = []
    k = 0
    print("Eigenfiltering...")
    for i in tqdm.trange(n_patches):
        # Convolute Filter with defective image and calculate energies
        PATCH_DEF = patches_def[i]
        k += 1
        defective_filtered_all = []
        defective_energies = []
        image = image_def * 0
        i = 0
        for filter in filters:
            defective_filtered = convolute_filter(PATCH_DEF, filter)
            defective_filtered_all.append(defective_filtered)
            energy = get_energy(defective_filtered, blur=blurring)
            defective_energies.append(energy)
            i += 1


        # fig, ax = plt.subplots(ncols=4)
        # ax[0].imshow(PATCH_DEF)
        # ax[1].imshow(filter)
        # ax[2].imshow(defective_filtered)
        # ax[3].imshow(energy)
        # plt.show()
        
        defective_energies = np.array(defective_energies)

        # Calculate Mahalanobis distance between good and defective energies.
        mh = mahalanobis_matrix(defective_energies, good_energies)

        # Set edges to zero TODO: why necessary? 
        nrows = int(np.sqrt(n_patches))
        mh[:edge, :] = 0.#np.mean(mh)
        mh[:, :edge] = 0.#np.mean(mh)
        mh[:, -edge:] = 0.#np.mean(mh)            
        mh[-edge:, :] = 0.#np.mean(mh)
        # if (not k%nrows==0):
        #     mh[:, -edge:] = 0            
        # if (not k//nrows==nrows):
        #     mh[-edge:, :] = 0
        mhs.append(mh)

    result = helpers.get_image_from_patches(image=image, patches=mhs, target_size=target_size, step=step)
    result = handle_edges(result, edgesize=edge, patchsize=target_size)
    print("Analysis done in {} seconds.".format(time.time() - starttime))
    return result


if __name__ == '__main__':
    data_directory = "../../data/cubic/defective/fold0/train/defective/images"
    # data_directory = "../../data/cubic/non_defective/fold0/train/nondefective/images"

    assert os.path.isdir(data_directory)
    paths = os.listdir(data_directory)
    paths.sort()

    print(paths)
    for path in paths:
        # Load image
        image_path = os.path.join(data_directory, path)
        image = helpers.get_image(image_path)
        var = get_variance_ratio(image, plot=True)
        print(var)

