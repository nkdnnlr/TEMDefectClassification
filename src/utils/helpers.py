import os
import tqdm
import shutil
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import cv2
# from keras.callbacks import Callback, BaseLogger
# import mlflow


def check_gpu():
    """
    Checks and prints the status of devices. Returns boolean with GPU availability
    :return:
    """
    import tensorflow as tf
    with tf.Session() as sess:
        devices = sess.list_devices()
        print("devices:")
        print(devices)

    print("tf build with CUDA: " + str(tf.test.is_built_with_cuda()))
    print("gpu available: " + str(tf.test.is_gpu_available()))
    print("CUDA gpu available: " + str(tf.test.is_gpu_available(cuda_only=True)))
    print("gpu device name: " + str(tf.test.gpu_device_name()))
    return tf.test.is_gpu_available()


def get_image(image_path):
    """
    Gets image from path
    :param image_path: path to image file
    :return: np.array
    """
    assert os.path.exists(image_path)
    original = cv2.imread(image_path, 0)
    return original


def preprocess(original):
    """
    Preprocess image
    :param original: original image
    :return:
    """
    preprocessed = original
    preprocessed = 255-(cv2.Laplacian(preprocessed, cv2.CV_64F, ksize=19))
    preprocessed = cv2.GaussianBlur(preprocessed, (5, 5), 0)
    return preprocessed


def get_patches(image, target_size=64, step=None, preprocess=False):
    """
    Crops image into square patches
    :param image_path: path to original image
    :param target_size: side length of patch
    :param step: Steps between patches. If None, step=target_size
    :param preprocess:
    :return: List of patches (2d-numpy arrays)
    """
    height, width = image.shape

    if step is None:
        step = target_size

    patches = []
    for y in range(0, height-target_size+1, step):
        for x in range(0, width-target_size+1, step):
            patch = image[y:y+target_size, x:x + target_size]
            patches.append(patch)
    #
    # for y in range(target_size, height+1, step):
    #     for x in range(target_size, width+1, step):
    #         patch = image[y - target_size:y, x - target_size:x]
    #         patches.append(patch)
    return patches


def load_image_cv2(image_path):
    """
    Loads image from path
    :param image_path: path to image file
    :return: np.array of OpenCV format
    """
    import cv2 as cv2
    assert os.path.exists(image_path)
    return cv2.imread(image_path, 0)


def count_files(directory_path):
    """
    Count all files in directory and subdirectories
    :param directory_path: Path to directory
    :return:
    """
    total = 0
    for root, dirs, files in os.walk(directory_path):
        total += len(files)
    return total


def reset_weights(model):
    """
    Resets weights of a keras model
    :param model:
    :return:
    """
    import keras.backend as K
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)


def get_graph(arr=None, connectivity=4):
    """
    Gets two planar 4- or 8-connected image graphs from binary image
    :param arr: binary ndarray, representing the image segments
    :param connectivity: either 4 or 8
    :return: G_0 and G_1: connected subgraphs with values 0 and with 1 respectively
             nx_pos: position dictionary for graph, following the networkx convention
    """
    import numpy as np
    import networkx as nx
    from contextlib import suppress

    G = nx.Graph()
    # Get nodes and positions
    pos = {}
    i = 0
    for (x, y), item in np.ndenumerate(arr):
        if item == 1:
            G.add_node(i)#, attr_dict={"val": 1})
            G.node[i]['val'] = 1

        else:# item == 0:
            G.add_node(i)#, atrr_dict={"val": 0})
            G.node[i]['val'] = 0
        # else:
        #     print("Unexpected Value")
        pos[i] = (x,y)
        i += 1
    inv_pos = {v: k for k, v in pos.items()}

    # Get edges
    i = 0
    for (x, y), item in np.ndenumerate(arr):
        # Check left.
        if x > 0:
            if arr[x-1, y] == item:
                j = inv_pos[(x-1, y)]
                # print(j)
                G.add_edge(i, j)
        # Check right.
        with suppress(IndexError):
            if arr[x+1, y] == item:
                j = inv_pos[(x+1, y)]
                # print(j)
                G.add_edge(i, j)
        # Check top.
        with suppress(IndexError):
            if arr[x, y+1] == item:
                j = inv_pos[(x, y+1)]
                # print(j)
                G.add_edge(i, j)
        # Check bottom.
        if y > 0:
            if arr[x, y-1] == item:
                j = inv_pos[(x, y-1)]
                # print(j)
                G.add_edge(i, j)

        if connectivity == 8:
            # Check left.
            if y > 0 and x > 0:
                with suppress(IndexError):
                    if arr[x - 1, y - 1] == item:
                        j = inv_pos[(x - 1, y - 1)]
                        # print(j)
                        G.add_edge(i, j)
            # Check right.
            # with suppress(IndexError):
            if x > 0:
                with suppress(IndexError):
                    if arr[x - 1, y + 1] == item:
                        j = inv_pos[(x - 1, y + 1)]
                        # print(j)
                        G.add_edge(i, j)
            # Check top.
            # with suppress(IndexError):
            if y > 0:
                with suppress(IndexError):
                    if arr[x + 1, y - 1] == item:
                        j = inv_pos[(x + 1, y - 1)]
                        # print(j)
                        G.add_edge(i, j)
            # Check bottom.
            # with suppress(IndexError):
            if True:
                with suppress(IndexError):
                    if arr[x + 1, y + 1] == item:
                        j = inv_pos[(x + 1, y + 1)]
                        # print(j)
                        G.add_edge(i, j)
        i += 1

    # Make subgraphs
    G_1 = G.subgraph([i for i in range(len(G)) if G.node[i]['val'] == 1])
    G_0 = G.subgraph([i for i in range(len(G)) if G.node[i]['val'] == 0])

    # Adjust positions to follow networkx-scheme
    height, width = arr.shape
    nx_pos = {}
    for key, value in pos.items():
        x, y = value
        nx_pos[key] = (y, width - x)

    return G_0, G_1, nx_pos


def localvariance_filter(image, size=20):
    """
    Convolves local variance filter with image
    :param image:
    :param size: kernel size
    :return:
    """
    import cv2

    wmean, wsqrmean = (cv2.boxFilter(x, -1, (size, size), borderType=cv2.BORDER_DEFAULT) for x in (image, image*image))
    return (wsqrmean - wmean*wmean)


def get_most_central_node(G):
    """
    Calculate the closeness centrality for all nodes of graph. Return maximum node
    :param G: graph
    :return: node with maximal closeness centrality
    """
    import networkx as nx
    closeness = nx.closeness_centrality(G)
    return max(closeness, key=closeness.get)


def draw_graphs(graphs, nx_pos, extra_nodes=None):
    """
    Draws node-graphs nicely
    :param graphs: list of graphs
    :param nx_pos: positions of nodes
    :param extra_nodes: extra nodes to add to plot
    :return:
    """
    import networkx as nx

    # colors = ['#b361b4', '#f4c06f']
    colors = ['#440154', '#fde724']
    # Draw graphs
    fig, ax = plt.subplots(figsize=(8, 8))
    i = 0
    for graph in graphs:
        nx.draw(graph, pos=nx_pos, node_color=colors[i], with_labels=False, ax=ax)
        i += 1
        # nx.draw(G_0, node_color='b', pos=nx_pos, with_labels=True)
        if extra_nodes is not None:
            nx.draw(graph.subgraph(extra_nodes), node_color='#78d0aa', pos=nx_pos, ax=ax)
    return fig, ax


def draw_image_patched(image, patchsize):
    """
    Draw image with patches denoted
    :param image:
    :param patchsize:
    :return:
    """
    width, height = image.shape
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")

    for pos in range(patchsize, width, patchsize):
        ax.plot([pos, pos], [0, height], 'k', linewidth=0.8)

    for pos in range(patchsize, height, patchsize):
        ax.plot([0, width], [pos, pos], 'k', linewidth=0.8)

    ax.set_xlim([0,width])
    ax.set_ylim([0,height])
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax

def map_scores(image, scores, target_size, step_size):
    """
    Visualizing results by creating a heatmap, showing defective areas
    :param image:
    :param scores:
    :return:
    """
    height, width = image.shape
    side = np.sqrt(len(scores))

    patch_size = target_size
    step = step_size

    result = np.zeros([height, width])
    i = 0
    for y in range(patch_size, height+1, step):
        for x in range(patch_size, width+1, step):
            result[y - patch_size:y, x - patch_size:x] += scores[i]
            i += 1
    return result


def get_image_from_patches(image, patches, target_size=64, step=None, preprocess=False, special_patch=None):
    """
    Stitch together patch to image.
    :param image:
    :param patches:
    :param target_size: side length of patch
    :param step: Steps between patches. If None, step=target_size
    :param preprocess:
    :return:
    """
    if special_patch is not None:
        s = special_patch
    else:
        s = -1
    height, width = image.shape

    if step is None:
        step = target_size

    i = 0
    for y in range(0, height-target_size+1, step):
        for x in range(0, width-target_size+1, step):
            if i == s:
                image[y:y + target_size, x:x + target_size] = patches[i]*0
            image[y:y+target_size, x:x + target_size] += patches[i]
            i += 1
            if i >= len(patches):
                return image
    return image


def plot_all_raw_images(nondefective, defective, output_dir):
    """
    Plot nondefective and defective images
    :param nondefective:
    :param defective:
    :param output_dir:
    :return:
    """
    import matplotlib.gridspec as gridspec

    parent_dir = nondefective
    plt.figure(figsize=(16, 28))
    gs1 = gridspec.GridSpec(7, 4)
    gs1.update(wspace=0.025, hspace=0.05)
    i = 0
    for file in os.listdir(parent_dir):
        if file.endswith(".tif"):
            # LOAD IMAGE AND PREPROCESS
            name = file[:-4]
            file = os.path.join(parent_dir, file)
            original = get_image(file)
            try:
                ax1 = plt.subplot(gs1[i])
                ax1.imshow(original, cmap='gray')
                ax1.set_xticks([])
                ax1.set_yticks([])
            except IndexError: continue
            i += 1
    plt.savefig(os.path.join(output_dir, "raw_nondefective"), bbox_inches='tight')
    plt.show()

    parent_dir = defective
    plt.figure(figsize=(16, 16))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0.025, hspace=0.05)
    i = 0
    for file in os.listdir(parent_dir):
        if file.endswith(".tif"):
            # LOAD IMAGE AND PREPROCESS
            name = file[:-4]
            file = os.path.join(parent_dir, file)
            original = get_image(file)

            try:
                ax1 = plt.subplot(gs1[i])
                ax1.imshow(original, cmap='gray')
                ax1.set_xticks([])
                ax1.set_yticks([])
            except IndexError: continue
            i += 1
    for j in range(i,16):
        ax1 = plt.subplot(gs1[j])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.axis('off')
    plt.savefig(os.path.join(output_dir, "raw_defective"), bbox_inches='tight')
    plt.show()

def add_bestpatch_to_img(image, patchsize, patch, color='gray'):
    """
    Adding best patch notation to image
    :param image:
    :param patchsize:
    :param patch:
    :param color:
    :return:
    """
    height, width = image.shape
    total = patch*patchsize
    start_x = (total // width)*patchsize
    start_y = (total % width)

    fig = plt.figure()
    plt.imshow(image, color)

    plt.plot([start_y, start_y, start_y+patchsize, start_y+patchsize, start_y],
             [start_x, start_x+patchsize, start_x+patchsize, start_x, start_x],
             'r')
    return fig

def make_publication_subplots(image, predict, segment, localvariance, patchsize, patchnr, maxfiltered, minlocalvariance, output_path=None):
    """
    Make nice graph for eigenfilter segmentation process.
    :param image:
    :param predict:
    :param segment:
    :param localvariance:
    :param patchsize:
    :param patchnr:
    :param maxfiltered:
    :param minlocalvariance:
    :param output_path:
    :return:
    """
    import matplotlib.gridspec as gridspec
    height, width = image.shape
    total = patchnr*patchsize
    start_x = (total // width)*patchsize
    start_y = (total % width)

    plt.figure(figsize=(32, 8))
    gs1 = gridspec.GridSpec(1, 4)
    gs1.update(wspace=0.05, hspace=0.05)

    ax0 = plt.subplot(gs1[0])
    ax0.imshow(image, cmap='gray')
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = plt.subplot(gs1[1])
    ax1.imshow(predict, cmap='viridis')
    ax1.plot([start_y, start_y, start_y+patchsize, start_y+patchsize, start_y],
             [start_x, start_x+patchsize, start_x+patchsize, start_x, start_x],
             'r')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = plt.subplot(gs1[2])
    ax2.imshow(segment, cmap='viridis', vmin=7, vmax=np.max((22, maxfiltered)))
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = plt.subplot(gs1[3])
    ax3.imshow(localvariance, cmap='inferno_r', vmin=minlocalvariance-300, vmax=minlocalvariance)
    ax3.set_xticks([])
    ax3.set_yticks([])

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    # plt.show()


def get_best_node_from_Kneighbors(G, k=10, connectivity=4):
    """
    For each node in a graph, build a weighted sum of all neighbors with respect to their closeness.
    :param G: networkx graph object
    :param k: maximal closeness
    :param connectivity: graph connectivity. Either 4 or 8.
    :return: node that maximizes (sum_i^k (n_i / ((i*4)^2 * 2^i) )) for n=neighbor
    """
    import networkx as nx
    import math

    neighbors = {}
    weighted_neighbors = {}
    for node in G.nodes():
        nn = nx.single_source_shortest_path_length(G, node, cutoff=k)
        neighbors[node] = []
        weighted_neighbors[node] = 0
        for i in range(1, k+1):
            occurence = sum(value == i for value in nn.values())
            neighbors[node].append(occurence)
            weighted_neighbors[node] += occurence/(math.factorial(i)*connectivity**i)
    best = max(weighted_neighbors, key=weighted_neighbors.get)
    return best


# class LossHistory(Callback):
#     """
#     Keras Callback that allows logging on MLFlow after each epoch end
#     """

#     def on_epoch_end(self, epoch, logs=None):
#         metrics = logs.keys()
#         for metric in metrics:
#             # print("metric: ", metric)
#             mlflow.log_metric(metric, logs.get(metric), step=epoch)
