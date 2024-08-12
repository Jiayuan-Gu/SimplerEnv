"""
Obtain dataset trajectory samples and save them for system identification.
"""

import argparse
import pickle

import numpy as np
from sapien.core import Pose
import tensorflow_datasets as tfds
from transforms3d.axangles import mat2axangle
from transforms3d.euler import euler2axangle, euler2mat, euler2quat
from transforms3d.quaternions import axangle2quat, mat2quat, quat2axangle, quat2mat

DATASETS = ["fractal20220817_data", "bridge"]


def dataset2path(dataset_name):
    if dataset_name == "robo_net":
        version = "1.0.0"
    elif dataset_name == "language_table":
        version = "0.0.1"
    else:
        version = "0.1.0"
    return f"gs://gresearch/robotics/{dataset_name}/{version}"


FRACTAL_CLIP_RATIO = [
    6 / 22,
    5 / 13,
    2 / 4,
    1 / 4,
    2 / 7,
    6 / 12,
    3 / 7,
    7 / 17,
    3 / 9,
    2 / 7,
    3 / 5,
    2 / 4,
    2 / 5,
    4 / 8,
    3 / 10,
    2 / 13,
    3 / 12,
    1 / 8,
    2 / 7,
    4 / 10,
    1 / 4,
    5 / 9,
    1 / 5,
    9 / 13,
    2 / 5,
    3 / 8,
    1.5 / 8,
    2.5 / 7,
    3 / 5,
    4 / 8,
    3 / 8,
    1 / 7,
    2 / 8,
    3 / 9,
    6 / 9,
    2 / 9,
    2 / 3,
    3 / 8,
    1.5 / 7,
    1.5 / 6,
    2 / 4,
    2 / 7,
    1 / 4,
    6 / 12,
    1.5 / 7,
    3 / 9,
    3 / 9,
    5 / 11,
    2 / 15,
    6 / 14,
]

FRACTAL_EVAL_CLIP_RATIO = [
    2 / 5,
    4 / 12,
    1.5 / 5,
    2.5 / 5,
    2 / 5,
    3 / 5,
    2 / 4,
    1 / 3,
    2 / 5,
    2.5 / 8,
]

if __name__ == "__main__":
    """
    python tools/sysid/prepare_sysid_dataset.py --save-path /home/xuanlin/Downloads/sysid_dataset.pkl --dataset-name fractal20220817_data
    python tools/sysid/prepare_sysid_dataset.py --save-path /home/xuanlin/Downloads/sysid_dataset_bridge.pkl --dataset-name bridge
    
    python tools/sysid/prepare_sysid_dataset.py --save-path sysid_logs/sysid_dataset.pkl --dataset-name fractal20220817_data
    python tools/sysid/prepare_sysid_dataset.py --save-path sysid_logs/sysid_dataset_eval_no_contact.pkl --dataset-name fractal20220817_data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default="/home/xuanlin/Downloads/sysid_dataset.pkl")
    parser.add_argument("--dataset-name", type=str, default="fractal20220817_data")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    assert dataset_name in DATASETS
    dset = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))

    dset = dset.as_dataset(split="train", read_config=tfds.ReadConfig(add_tfds_id=True))
    dset_iter = iter(dset)
    iter_episode_id = -1
    if dataset_name == "fractal20220817_data":
        # episode_ids = [2, 4, 5, 9, 11, 805, 1257, 1495, 1539, 1991, 2398, 3289]
        # episode_ids = [
        #     2,
        #     4,
        #     5,
        #     9,
        #     11,
        #     14,
        #     28,
        #     34,
        #     36,
        #     37,
        #     38,
        #     805,
        #     1257,
        #     1495,
        #     1539,
        #     1991,
        #     2398,
        #     3289,
        # ]
        # episode_ids = list(range(50))
        episode_ids = [
            50,
            52,
            53,
            805,
            1257,
            1495,
            1539,
            1991,
            2398,
            3289,
        ]

    elif dataset_name == "bridge":
        episode_ids = list(range(12))
    else:
        raise NotImplementedError()

    save = []
    while iter_episode_id <= max(episode_ids):
        iter_episode_id += 1
        episode = next(dset_iter)
        if iter_episode_id not in episode_ids:
            continue
        
        print(f"Processing episode {iter_episode_id}")
        
        # # Clip the episode to contact-free
        # clip_ratio = FRACTAL_CLIP_RATIO[iter_episode_id]
        clip_ratio = FRACTAL_EVAL_CLIP_RATIO[episode_ids.index(iter_episode_id)]
        clip_length = int(clip_ratio * len(episode["steps"]))

        to_save = []
        episode_steps = list(episode["steps"])

        # # Skip fridge and drawer tasks
        # instruction = episode_steps[0]["observation"]["natural_language_instruction"].numpy().decode("utf-8")
        # # print(instruction)
        # if "fridge" in instruction or ("close" in instruction and "drawer" in instruction) or ("open" in instruction and "drawer" in instruction):
        #     print(f"Skipping episode {iter_episode_id} with instruction {instruction}")
        #     continue

        for j, episode_step in enumerate(episode_steps):
            if dataset_name == "fractal20220817_data":
                if j == 0:
                    continue  # skip the first step since during the real data collection process, its action might not be reach the robot in time and be executed by the robot
                if j >= clip_length:
                    break
                # if j >= 50:
                #     print(f"Skipping step {j} in episode {iter_episode_id}")
                #     break
                base_pose_tool_reached = episode_step["observation"]["base_pose_tool_reached"]
                base_pose_tool_reached = np.concatenate(
                    [
                        base_pose_tool_reached[:3],
                        base_pose_tool_reached[-1:],
                        base_pose_tool_reached[3:-1],
                    ]
                )  # [xyz, quat(wxyz)]
                save_episode_step = {
                    "base_pose_tool_reached": np.array(
                        base_pose_tool_reached, dtype=np.float64
                    ),  # reached tool pose under the robot base frame
                    "action_world_vector": np.array(episode_step["action"]["world_vector"], dtype=np.float64),
                    "action_rotation_delta": np.array(episode_step["action"]["rotation_delta"], dtype=np.float64),
                    # 'action_gripper': np.array(episode_step['action']['gripper_closedness_action'], dtype=np.float64), # 1=close; -1=open
                }
            elif dataset_name == "bridge":
                mat_transform = np.array(
                    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
                    dtype=np.float64,
                )
                base_pose_tool_reached = Pose(
                    p=episode_step["observation"]["state"][:3],
                    q=mat2quat(
                        euler2mat(
                            *np.array(
                                episode_step["observation"]["state"][3:6],
                                dtype=np.float64,
                            )
                        )
                        @ mat_transform
                    ),
                )
                save_episode_step = {
                    "base_pose_tool_reached": np.concatenate(
                        [
                            np.array(base_pose_tool_reached.p, dtype=np.float64),
                            np.array(base_pose_tool_reached.q, dtype=np.float64),
                        ]
                    ),  # reached tool pose under the robot base frame, [xyz, quat(wxyz)]
                    "action_world_vector": np.array(episode_step["action"]["world_vector"], dtype=np.float64),
                    "action_rotation_delta": np.array(episode_step["action"]["rotation_delta"], dtype=np.float64),
                    # 'action_gripper': np.array(2.0 * (np.array(episode_step['action']['open_gripper'])[None]) - 1.0, dtype=np.float64), # 1=open; -1=close
                }
            else:
                raise NotImplementedError()
            to_save.append(save_episode_step)
        save.append(to_save)

    with open(args.save_path, "wb") as f:
        pickle.dump(save, f)
