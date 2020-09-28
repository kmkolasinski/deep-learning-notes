from typing import List, Tuple

import glm
import matplotlib.pyplot as plt
import numpy as np
import cv2
import plotly
import plotly.graph_objs as go


def plot_glm_vec2(t_points: np.ndarray, color = "r", image_size: Tuple[int, int] = (300, 300), figsize: Tuple[int, int] = (5, 5), **kwargs):
    x = np.array([p[0] for p in t_points if p[2] > 0])
    y = np.array([p[1] for p in t_points if p[2] > 0])
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.scatter(x, y, color=color, **kwargs)
    plt.xlim([0, image_size[0]])
    plt.ylim([0, image_size[1]])


def match_in_viewport(
        prev_points: np.ndarray,
        new_points: np.ndarray,
        image_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    is_in_viewport = lambda p: \
        np.alltrue(p[:2] >= (0, 0)) \
        and np.alltrue(p[:2] < image_size)

    prev_points_matched = []
    new_points_matched = []
    indices = []
    for k, (pp, cp) in enumerate(zip(prev_points, new_points)):
        if is_in_viewport(pp) and is_in_viewport(cp):
            prev_points_matched += [pp[:2]]
            new_points_matched += [cp[:2]]
            indices.append(k)

    prev_points_matched = np.array(prev_points_matched)
    new_points_matched = np.array(new_points_matched)

    return np.array(indices), prev_points_matched, new_points_matched


def plot_3d_points(data: List[np.ndarray], mode="markers", size=3):
    # Configure the trace.
    traces = []
    for points in data:
        trace = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode=mode,
            marker={
                'size': size,
                'opacity': 0.8,
            }
        )
        traces.append(trace)

    layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
    plot_figure = go.Figure(data=traces, layout=layout)
    plotly.offline.iplot(plot_figure)


from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform


def recover_pose(prev_points: np.ndarray, new_points: np.ndarray, camera_mat: np.ndarray, distCoeff = None):

    E0, mask0 = cv2.findEssentialMat(prev_points, new_points,
                                   cameraMatrix=camera_mat, method=cv2.LMEDS,
                                   prob=0.99, threshold=0.1)

    # model, mask = ransac((prev_points, new_points),
    #                         EssentialMatrixTransform, min_samples=8,
    #                         residual_threshold=1, max_trials=1000)
    #
    # E = np.array(model.params)
    # mask = (mask * 255).astype("uint8")
    E = E0
    mask = mask0
    _, cur_R, cur_t, mask = cv2.recoverPose(
        E, prev_points, new_points,
        cameraMatrix=camera_mat, mask=mask
    )
    # print(cur_t)
    prev_mat = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])

    proj_mat0 = camera_mat.dot(prev_mat)
    new_mat = np.hstack((cur_R, cur_t))
    proj_mat1 = camera_mat.dot(new_mat)

    p0_points = prev_points.T.copy()
    p1_points = new_points.T.copy()
    points_3d = cv2.triangulatePoints(proj_mat0, proj_mat1, p0_points, p1_points).T
    points_3d = points_3d[:, :3] / points_3d[:, -1:]

    return new_mat, points_3d
