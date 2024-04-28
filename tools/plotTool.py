import matplotlib.pyplot as plt
import numpy as np


def plot_result(pred_poses, targ_poses, data_set):
    # this function is original from https://github.com/NVlabs/geomapnet
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    # plot on the figure object
    ss = max(1, int(len(data_set) / 1000))  # 100 for stairs
    # scatter the points and draw connecting line
    x = np.vstack((pred_poses[::ss, 0].T, targ_poses[::ss, 0].T))
    y = np.vstack((pred_poses[::ss, 1].T, targ_poses[::ss, 1].T))
    z = np.vstack((pred_poses[::ss, 2].T, targ_poses[::ss, 2].T))
    for xx, yy, zz in zip(x.T, y.T, z.T):
      ax.plot(xx, yy, zs=zz, c='gray', alpha=0.6)
    ax.scatter(x[0, :], y[0, :], zs=z[0, :], c='r', depthshade=0, alpha=0.8)
    ax.scatter(x[1, :], y[1, :], zs=z[1, :], c='g', depthshade=0, alpha=0.8)
    ax.view_init(azim=119, elev=13)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    plt.show()

