import matplotlib.pyplot as plt

def save_scene(file_path, scene_obs, dpi=300):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(scene_obs, aspect='auto')
    fig.savefig(file_path, dpi= dpi)
    