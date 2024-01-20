import numpy as np
import os, cv2
import mediapy as media

from real2sim.octo.octo_model import OctoInference
from real2sim.utils.visualization import write_video
from real2sim.utils.env.env_builder import build_maniskill2_env
from sapien.core import Pose

def main(input_video, impainting_img_path, instruction,
         gt_tcp_pose_at_robot_base=None, camera='3rd_view_camera',
         model_type='octo-base',
         control_freq=5):
    
    # Create environment
    env = build_maniskill2_env(
        'PickCube-v0',
        control_mode='arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos',
        obs_mode='rgbd',
        robot='widowx',
        sim_freq=500,
        max_episode_steps=90,
        control_freq=control_freq,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=impainting_img_path,
        rgb_overlay_cameras=[camera],
    )
    print(instruction)
    
    # Reset and initialize environment
    predicted_actions = []
    images = []
    
    obs, _ = env.reset()
    env.agent.robot.set_pose(Pose([0, 0, 1]))
    
    if gt_tcp_pose_at_robot_base is not None:
        controller = env.agent.controller.controllers['arm']
        cur_qpos = env.agent.robot.get_qpos()
        init_arm_qpos = controller.compute_ik(gt_tcp_pose_at_robot_base)
        cur_qpos[controller.joint_indices] = init_arm_qpos
        env.agent.reset(cur_qpos)
    
    image = (env.get_obs()["image"][camera]['Color'][..., :3] * 255).astype(np.uint8)
    images.append(image)

    octo_model = OctoInference(model_type, action_scale=1.0)
    # Reset Octo model
    octo_model.reset(instruction)

    timestep = 0
    truncated = False
    # Step the environment
    if input_video is not None:
        loop_criterion = lambda timestep: timestep < len(input_video) - 1
    else:
        loop_criterion = lambda timestep: not truncated
    while loop_criterion(timestep):
        if input_video is not None:
            raw_action, action = octo_model.step(input_video[timestep])
        else:
            raw_action, action = octo_model.step(image)
        predicted_actions.append(raw_action)
        print(timestep, raw_action)
        
        obs, reward, terminated, truncated, info = env.step(
            np.concatenate(
                [action['world_vector'], 
                action['rot_axangle'],
                action['gripper']
                ]
            )
        )
        
        image = obs['image'][camera]['rgb']
        images.append(image)
        timestep += 1

    octo_model.visualize_epoch(predicted_actions, images, save_path='/home/xuanlin/Downloads/debug_octo_inference.png')
    
    if input_video is not None:
        for i in range(len(images)):
            images[i] = np.concatenate(
                [cv2.resize(images[i], (input_video[i].shape[1], input_video[i].shape[0])), 
                input_video[i]], 
                axis=1
            )
    video_path = f'/home/xuanlin/Downloads/debug_octo_inference.mp4'
    write_video(video_path, images, fps=5)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['DISPLAY'] = ''
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    # mp4_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/debug/bridge_real_1.mp4'
    # # mp4_path = None
    # impainting_img_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/debug/bridge_real_1_cleanup.png'
    # instruction = 'Place the can to the left of the pot.'
    # gt_tcp_pose_at_robot_base = Pose([0.298068, -0.114657, 0.10782], [0.750753, 0.115962, 0.642171, -0.102661])
    # camera = '3rd_view_camera_bridge'
    
    # mp4_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/debug/bridge_real_2.mp4'
    # impainting_img_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/debug/bridge_real_2_cleanup.png'
    # instruction = 'Pick up the bowl.'
    # instruction = 'Move the kadai and place it at the right edge of the table.' # doesn't understand the instruction...
    # gt_tcp_pose_at_robot_base = Pose([0.350166, -0.0610973, 0.157404], [0.73995, -0.377095, 0.438111, 0.343994])
    # camera = '3rd_view_camera_bridge'
    
    mp4_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/debug/bridge_real_eval_spoononcloth_1.mp4'
    impainting_img_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/debug/bridge_real_eval_spoononcloth_1_cleanup.png'
    instruction = 'put the spoon on the towel'
    gt_tcp_pose_at_robot_base = None
    camera = '3rd_view_camera'
    
    # mp4_path = None
    # impainting_img_path = '/home/xuanlin/Real2Sim/ManiSkill2_real2sim/data/debug/bridge_real_stackcube_2.png'
    # instruction = 'stack the green block on the yellow block'
    # gt_tcp_pose_at_robot_base = None
    # camera = '3rd_view_camera'
    
    if mp4_path is not None:
        input_video = media.read_video(mp4_path)
    else:
        input_video = None
    
    main(input_video, impainting_img_path, instruction, gt_tcp_pose_at_robot_base, camera)