import os
import math
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

        
def save_video_fig(
        fig_id: plt.figure,
        fig_name: str, 
        fig_dir: str = '', 
        parent_dir_name: str = '',
        tight_layout: bool = True
):
    # create the directory if it does not exist
    img_dir = os.path.join(parent_dir_name, fig_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    path = os.path.join(img_dir, fig_name + ".png")
    print("Saving figure", fig_name)
    if tight_layout:
        fig_id.tight_layout()
    fig_id.savefig(path, format='png', dpi=300)


def get_video_names(
    metadata: dict
) -> Tuple[str, str] :
    video_name = metadata['video_name'].numpy().decode('UTF-8')
    print("Video: ", video_name)
    video_type = metadata['video_type'].numpy().decode('UTF-8')
    print("Video type: ", video_type)
    return video_name, video_type


def get_scale_offset(
    metadata: dict, 
    imgdata: str = 'forward_flow',
) -> Tuple[float, float] : 
    i_range = metadata[imgdata + '_range'].numpy()
    print("{0} range {1} to {2}".format(imgdata,
                                        i_range[0],
                                        i_range[1]))
    i_scale =  ( i_range[1] - i_range[0] ) / 65535.0
    i_offset = i_range[0]
    return i_scale, i_offset


def flow_quiver(
    ax : plt.Axes,
    flow : np.ndarray,
    x_mesh : np.ndarray = np.empty(1),
    y_mesh : np.ndarray = np.empty(1),        
    n_arrows: int = 32,
    img_size: int = 256
) :
    if (x_mesh.shape[0] < math.floor(img_size/n_arrows) or
            y_mesh.shape[0] < math.floor(img_size/n_arrows)) :
        x_mesh, y_mesh = np.meshgrid(np.linspace(0, img_size, n_arrows),
                           np.linspace(0, img_size, n_arrows))

    # quiver has origin at lower left corner, offsets are x=col, y=-row
    res_quiver = ax.quiver(
        x_mesh, -y_mesh,
        flow[0: img_size: int(img_size / n_arrows),
         0: img_size: int(img_size / n_arrows),
         1],
        -flow[0: img_size: int(img_size / n_arrows),
         0: img_size: int(img_size / n_arrows),
         0],
         angles='xy')    
    return res_quiver, x_mesh, y_mesh


def get_fg_bg_mask( 
            segmentation : np.ndarray,
            mask : np.ndarray 
) -> Tuple[ np.ndarray, np.ndarray ]:
    bg_mask = np.logical_and((segmentation == 0), mask)  # background id is 0
    fg_mask = np.logical_and((segmentation > 0), mask)
    return fg_mask, bg_mask
