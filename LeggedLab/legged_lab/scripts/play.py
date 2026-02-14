import os
import copy
import torch
from legged_lab.utils import task_registry

import argparse

from isaaclab.app import AppLauncher
from rsl_rl.runners import OnPolicyRunner
# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record video in headless mode")
parser.add_argument("--video_length", type=int, default=500, help="Number of steps to record")
parser.add_argument("--video_fps", type=int, default=30, help="FPS for the recorded video")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# enable cameras if recording video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path


class PolicyExporterStudent(torch.nn.Module):
    def __init__(self, policy, student_encoder, num_obs, num_history_obs, command_dim):
        super().__init__()
        self.policy = copy.deepcopy(policy)
        self.student_encoder = copy.deepcopy(student_encoder)
        self.num_obs = num_obs
        self.num_history_obs = num_history_obs
        self.command_dim = command_dim

    def forward(self, input):
        current_obs = input[:, -self.num_obs:]
        history_obs = input[:, :]
        obs = current_obs[:, :-self.command_dim]
        latent = self.student_encoder.act_inference(history_obs)
        commands = current_obs[:, -self.command_dim:]
        actor_obs = torch.cat((obs, latent, commands), dim=-1)
        return self.policy.act_inference(actor_obs)

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        self.to('cpu')

        # path_pt = os.path.join(path, 'policy.pt')
        # traced_script_module = torch.jit.script(self)
        # traced_script_module.save(path_pt)
        # policy_jit_model = torch.jit.load(path_pt)
        # #set the model to evalution mode
        # policy_jit_model.eval()

        # creat a fake input to the model
        test_input_tensor = torch.randn(1, self.num_history_obs)

        #specify the path and name of the output onnx model
        path_onnx = os.path.join(path, 'policy.onnx')

        #export the onnx model
        torch.onnx.export(self,      # policy_jit_model
                        test_input_tensor,       
                        path_onnx,   # params below can be ignored
                        export_params=True,   
                        opset_version=11,     
                        do_constant_folding=True,  
                        input_names=['input'],    
                        output_names=['output'],  
                        )


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robot.enable = False
    env_cfg.scene.max_episode_length_s = 40.
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.ranges.lin_vel_x = (0.5, 0.5)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.heading = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z=(0.0, 0.0)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    from legged_lab.terrains import PlANE_TERRAINS_CFG
    env_cfg.scene.terrain_generator = PlANE_TERRAINS_CFG
    # env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    history_obs, extras = env.get_observations()
    critic_obs = extras["observations"].get("critic", history_obs)
    num_one_step_obs = history_obs.size(-1) // env_cfg.robot.actor_obs_history_length
    obs = history_obs[:, -num_one_step_obs:]
    command_dim = 3

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    policy, teacher_encoder, student_encoder = runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(os.path.dirname(resume_path), '../1/exported')
        exporter = PolicyExporterStudent(policy, student_encoder, obs.size(-1), history_obs.size(-1), command_dim)
        exporter.export(path)

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard
        keyboard = Keyboard(env)  # noqa:F841

    # set up video recording
    if args_cli.video:
        import numpy as np
        import omni.replicator.core as rep

        # create a camera at a good viewpoint
        camera_path = "/World/play_camera"
        camera = rep.create.camera(position=(3.0, 3.0, 2.5), look_at=(0.0, 0.0, 0.0))
        rp = rep.create.render_product(camera, (1280, 720))
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb_annot.attach([rp])

        # warm up rendering
        for _ in range(5):
            env.sim.render()

        frames = []
        print(f"[INFO] Recording video for {args_cli.video_length} steps...")

    step_count = 0
    while simulation_app.is_running():

        with torch.inference_mode():
            latent = teacher_encoder.act_inference(critic_obs[:, :-command_dim])
            commands = obs[:, -command_dim:]
            actor_obs = torch.cat((obs[:, 0:-command_dim], latent, commands), dim=-1)
            actions = policy.act_inference(actor_obs.detach())

            history_obs, _, _, infos = env.step(actions)
            critic_obs = infos["observations"]["critic"]
            obs = history_obs[:, -num_one_step_obs:]

        # capture frames for video
        if args_cli.video:
            env.sim.render()
            data = rgb_annot.get_data()
            if data is not None and data.shape[0] > 0:
                frames.append(np.array(data[:, :, :3]))

            step_count += 1
            if step_count >= args_cli.video_length:
                # save video
                video_dir = os.path.join(log_dir, "videos")
                os.makedirs(video_dir, exist_ok=True)
                video_path = os.path.join(video_dir, "play.mp4")

                import imageio
                writer = imageio.get_writer(video_path, fps=args_cli.video_fps)
                for frame in frames:
                    writer.append_data(frame)
                writer.close()
                print(f"[INFO] Video saved to: {video_path} ({len(frames)} frames)")
                break


if __name__ == '__main__':
    EXPORT_POLICY = False
    play()
    simulation_app.close()
