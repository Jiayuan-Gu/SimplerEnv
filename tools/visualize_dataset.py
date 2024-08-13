import numpy as np
import tensorflow_datasets as tfds

from simpler_env.utils.visualization import write_video

DATASETS = ["fractal20220817_data", "bridge"]


def dataset2path(dataset_name):
    if dataset_name == "robo_net":
        version = "1.0.0"
    elif dataset_name == "language_table":
        version = "0.0.1"
    else:
        version = "0.1.0"
    return f"gs://gresearch/robotics/{dataset_name}/{version}"


# FRACTAL_CLIP_RATIO = [
#     6 / 22,
#     5 / 13,
#     2 / 4,
#     1 / 4,
#     2 / 7,
#     6 / 12,
#     3 / 7,
#     7 / 17,
#     3 / 9,
#     2 / 7,
#     3 / 5,
#     2 / 4,
#     2 / 5,
#     4 / 8,
#     3 / 10,
#     2 / 13,
#     3 / 12,
#     1 / 8,
#     2 / 7,
#     4 / 10,
#     1 / 4,
#     5 / 9,
#     1 / 5,
#     9 / 13,
#     2 / 5,
#     3 / 8,
#     1.5 / 8,
#     2.5 / 7,
#     3 / 5,
#     4 / 8,
#     3 / 8,
#     1 / 7,
#     2 / 8,
#     3 / 9,
#     6 / 9,
#     2 / 9,
#     2 / 3,
#     3 / 8,
#     1.5 / 7,
#     1.5 / 6,
#     2 / 4,
#     2 / 7,
#     1 / 4,
#     6 / 12,
#     1.5 / 7,
#     3 / 9,
#     3 / 9,
#     5 / 11,
#     2 / 15,
#     6 / 14,
# ]

# EPISODE_IDS = [
#     50,
#     # 51,
#     52,
#     53,
#     805,
#     1257,
#     1495,
#     1539,
#     1991,
#     2398,
#     3289,
# ]

# FRACTAL_CLIP_RATIO = [
#     2 / 5,
#     4 / 12,
#     1.5 / 5,
#     2.5 / 5,
#     2 / 5,
#     3 / 5,
#     2 / 4,
#     1 / 3,
#     2 / 5,
#     2.5 / 8,
# ]

EPISODE_IDS = list(range(100, 152))
FRACTAL_CLIP_RATIO = [
    None,
    1 / 5,
    3 / 10,
    3 / 5,
    3 / 9,
    2 / 8,
    3 / 6,
    2.5 / 5,
    2 / 8,
    2 / 14,
    3 / 9,
    2 / 7,
    3 / 13,
    3 / 9,
    1 / 4,
    1.5 / 7,
    1.5 / 5,
    4 / 12,
    2 / 5,
    1 / 6,
    4 / 10,
    1 / 4,
    1.5 / 10,
    3 / 8,
    0.5 / 4,
    None,
    3 / 7,
    3 / 8,
    2.5 / 5,
    2 / 6,
    1 / 3,
    2 / 8,
    2 / 8,
    3.5 / 10,
    1 / 4,
    2 / 7,
    2 / 6,
    2 / 4,
    1.5 / 5,
    3 / 7,
    2 / 7,
    1 / 2,
    1 / 8,
    3 / 6,
    3 / 12,
    3 / 9,
    6 / 14,
    1 / 8,
    3 / 16,
    2 / 8,
    2 / 6,
    1 / 5,
]
EPISODE_IDS = EPISODE_IDS[1:25] + EPISODE_IDS[26:]
FRACTAL_CLIP_RATIO = FRACTAL_CLIP_RATIO[1:25] + FRACTAL_CLIP_RATIO[26:]


if __name__ == "__main__":
    dataset_name = DATASETS[0]
    dset = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))

    # dset = dset.as_dataset(split="train[:50]", read_config=tfds.ReadConfig(add_tfds_id=True))
    # dset = dset.as_dataset(split="train[:4000]", read_config=tfds.ReadConfig(add_tfds_id=True))
    dset = dset.as_dataset(split="train[:160]", read_config=tfds.ReadConfig(add_tfds_id=True))
    dset = list(dset)
    for i, episode in enumerate(dset):
        if i not in EPISODE_IDS:
            continue;
        if i > EPISODE_IDS[-1]:
            break;
        gt_images = []
        episode_steps = list(episode["steps"])
        # for j in range(len(episode_steps) - 1):
        # for j in range(0, int(FRACTAL_CLIP_RATIO[i] * len(episode_steps))):
        for j in range(0, int(FRACTAL_CLIP_RATIO[EPISODE_IDS.index(i)] * len(episode_steps))):
            gt_images.append(episode_steps[j]["observation"]["image"])
        write_video(f"{dataset_name}_vis/{i}_gt.mp4", gt_images, fps=5)
        # write_video(f"{dataset_name}_vis_full/{i}_gt.mp4", gt_images, fps=5)
        # write_video(f"{dataset_name}_vis_eval/{i}_gt.mp4", gt_images, fps=5)

        # from matplotlib import pyplot as plt

        # images = gt_images
        # ACTION_DIM_LABELS = ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'grasp']

        # img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # # set up plt figure
        # figure_layout = [
        #     ['image'] * len(ACTION_DIM_LABELS),
        #     ACTION_DIM_LABELS
        # ]
        # plt.rcParams.update({'font.size': 12})
        # fig, axs = plt.subplot_mosaic(figure_layout)
        # fig.set_size_inches([45, 10])

        # # plot actions
        # pred_actions = np.array([np.concatenate([episode_step['action']['world_vector'], episode_step['action']['rotation_delta'], episode_step['action']['open_gripper'][None]], axis=-1) for episode_step in episode_steps])
        # for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
        #     # actions have batch, horizon, dim, in this example we just take the first action for simplicity
        #     axs[action_label].plot(pred_actions[:, action_dim], label='predicted action')
        #     axs[action_label].set_title(action_label)
        #     axs[action_label].set_xlabel('Time in one episode')

        # axs['image'].imshow(img_strip)
        # axs['image'].set_xlabel('Time in one episode (subsampled)')
        # plt.legend()
        # plt.show()
