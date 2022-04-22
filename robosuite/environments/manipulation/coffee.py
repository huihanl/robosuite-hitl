from collections import OrderedDict
import numpy as np

from robosuite.utils.mjcf_utils import CustomMaterial

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import CoffeeMachinePodObject, CoffeeMachineObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.observables import Observable, sensor


class Coffee(SingleArmEnv):
    """
    This class corresponds to the coffee task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action):
        """
        Reward function for the task.

        Dense reward: TODO

        The sparse reward only consists of the threading component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.0 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # use a shaping reward
        if self.reward_shaping:
            pass

        if self.reward_scale is not None:
            reward *= self.reward_scale

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        self.coffee_pod = CoffeeMachinePodObject(name="coffee_pod")
        self.coffee_machine = CoffeeMachineObject(name="coffee_machine")
        objects = [self.coffee_pod, self.coffee_machine]

        # Create placement initializer
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CoffeeMachineSampler",
                mujoco_objects=self.coffee_machine,
                x_range=(0.0, 0.0),
                y_range=(-0.1, -0.1),
                rotation=(-np.pi / 6., -np.pi / 6.),
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CoffeePodSampler",
                mujoco_objects=self.coffee_pod,
                x_range=(-0.13, -0.07),
                y_range=(0.17, 0.23),
                rotation=(0.0, 0.0),
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.,
            )
        )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references for this env
        self.obj_body_id = dict(
            coffee_pod=self.sim.model.body_name2id(self.coffee_pod.root_body),
            coffee_machine=self.sim.model.body_name2id(self.coffee_machine.root_body),
            coffee_pod_holder=self.sim.model.body_name2id("coffee_machine_pod_holder_root"),
            coffee_machine_lid=self.sim.model.body_name2id("coffee_machine_lid_main"),
        )
        self.hinge_qpos_addr = self.sim.model.get_joint_qpos_addr("coffee_machine_lid_main_joint0")

        # for checking contact (used in reward function, and potentially observation space)
        self.pod_geom_id = self.sim.model.geom_name2id("coffee_pod_g0")
        self.lid_geom_id = self.sim.model.geom_name2id("coffee_machine_lid_g0")
        pod_holder_geom_names = ["coffee_machine_pod_holder_cup_body_hc_{}".format(i) for i in range(64)]
        self.pod_holder_geom_ids = [self.sim.model.geom_name2id(x) for x in pod_holder_geom_names]

        # size of bounding box for pod holder
        self.pod_holder_size = self.coffee_machine.pod_holder_size

        # size of bounding box for pod
        self.pod_size = self.coffee_pod.get_bounding_box_size()

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Always reset the hinge joint position
        self.sim.data.qpos[self.hinge_qpos_addr] = 2. * np.pi / 3.
        self.sim.forward()

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # for conversion to relative gripper frame
            @sensor(modality=modality)
            def world_pose_in_gripper(obs_cache):
                return T.pose_inv(T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))) if\
                    f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache else np.eye(4)
            sensors = [world_pose_in_gripper]
            names = ["world_pose_in_gripper"]
            actives = [False]

            # add ground-truth poses (absolute and relative to eef) for all objects
            for obj_name in self.obj_body_id:
                obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj_name, modality=modality)
                sensors += obj_sensors
                names += obj_sensor_names
                actives += [True] * len(obj_sensors)

            # add hinge angle of lid
            @sensor(modality=modality)
            def hinge_angle(obs_cache):
                return np.array([self.sim.data.qpos[self.hinge_qpos_addr]])
            sensors += [hinge_angle]
            names += ["hinge_angle"]
            actives += [True]

            # Create observables
            for name, s, active in zip(names, sensors, actives):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    active=active,
                )

        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """

        ### TODO: this was stolen from pick-place - do we want to move this into utils to share it? ###
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any([name not in obs_cache for name in
                    [f"{obj_name}_pos", f"{obj_name}_quat", "world_pose_in_gripper"]]):
                return np.zeros(3)
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return obs_cache[f"{obj_name}_to_{pf}eef_quat"] if \
                f"{obj_name}_to_{pf}eef_quat" in obs_cache else np.zeros(4)

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [f"{obj_name}_pos", f"{obj_name}_quat", f"{obj_name}_to_{pf}eef_pos", f"{obj_name}_to_{pf}eef_quat"]

        return sensors, names

    def _check_success(self):
        """
        Check if task is complete.
        """
        success = {}

        # lid should be closed (angle should be less than 5 degrees)
        hinge_tolerance = 15. * np.pi / 180. 
        hinge_angle = self.sim.data.qpos[self.hinge_qpos_addr]
        lid_check = (hinge_angle < hinge_tolerance)

        # pod should be in pod holder
        pod_holder_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod_holder"]])
        pod_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod"]])
        pod_check = True
        pod_horz_check = True

        # center of pod cannot be more than the difference of radii away from the center of pod holder
        r_diff = self.pod_holder_size[0] - self.pod_size[0]
        if np.linalg.norm(pod_pos[:2] - pod_holder_pos[:2]) > r_diff:
            pod_check = False
            pod_horz_check = False

        # make sure vertical pod dimension is above pod holder lower bound and below the lid lower bound
        lid_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_machine_lid"]])
        z_lim_low = pod_holder_pos[2] - self.pod_holder_size[2]
        z_lim_high = lid_pos[2] - self.coffee_machine.lid_size[2]
        if (pod_pos[2] - self.pod_size[2] < z_lim_low) or (pod_pos[2] + self.pod_size[2] > z_lim_high):
            pod_check = False

        success["task"] = lid_check and pod_check

        # partial task metrics below

        # for pod insertion check, just check that bottom of pod is within some tolerance of bottom of container
        pod_insertion_z_tolerance = 0.02
        pod_z_check = (pod_pos[2] - self.pod_size[2] > z_lim_low) and (pod_pos[2] - self.pod_size[2] < z_lim_low + pod_insertion_z_tolerance)
        success["insertion"] = pod_horz_check and pod_z_check

        # pod grasp check
        success["grasp"] = self._check_pod_is_grasped()

        # check is True if the pod is on / near the rim of the pod holder
        rim_horz_tolerance = 0.03
        rim_horz_check = (np.linalg.norm(pod_pos[:2] - pod_holder_pos[:2]) < rim_horz_tolerance)

        rim_vert_tolerance = 0.026
        rim_vert_length = pod_pos[2] - pod_holder_pos[2] - self.pod_holder_size[2]
        rim_vert_check = (rim_vert_length < rim_vert_tolerance) and (rim_vert_length > 0.)
        success["rim"] = rim_horz_check and rim_vert_check

        return success

    def _check_pod_is_grasped(self):
        """
        check if pod is grasped by robot
        """
        return self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=[g for g in self.coffee_pod.contact_geoms]
        )

    def _check_pod_and_pod_holder_contact(self):
        """
        check if pod is in contact with the container
        """
        pod_and_pod_holder_contact = False
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if(
                ((contact.geom1 == self.pod_geom_id) and (contact.geom2 in self.pod_holder_geom_ids)) or
                ((contact.geom2 == self.pod_geom_id) and (contact.geom1 in self.pod_holder_geom_ids))
            ):
                pod_and_pod_holder_contact = True
                break
        return pod_and_pod_holder_contact

    def _check_pod_on_rim(self):
        """
        check if pod is on pod container rim and not being inserted properly (for reward check)
        """
        pod_holder_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod_holder"]])
        pod_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod"]])

        # check if pod is in contact with the container
        pod_and_pod_holder_contact = self._check_pod_and_pod_holder_contact()

        # check that pod vertical position is not too low or too high
        rim_vert_tolerance_1 = 0.022
        rim_vert_tolerance_2 = 0.026
        rim_vert_length = pod_pos[2] - pod_holder_pos[2] - self.pod_holder_size[2]
        rim_vert_check = (rim_vert_length > rim_vert_tolerance_1) and (rim_vert_length < rim_vert_tolerance_2)

        return (pod_and_pod_holder_contact and rim_vert_check)

    def _check_pod_being_inserted(self):
        """
        check if robot is in the process of inserting the pod into the container
        """
        pod_holder_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod_holder"]])
        pod_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod"]])

        rim_horz_tolerance = 0.005
        rim_horz_check = (np.linalg.norm(pod_pos[:2] - pod_holder_pos[:2]) < rim_horz_tolerance)

        rim_vert_tolerance_1 = -0.01
        rim_vert_tolerance_2 = 0.023
        rim_vert_length = pod_pos[2] - pod_holder_pos[2] - self.pod_holder_size[2]
        rim_vert_check = (rim_vert_length < rim_vert_tolerance_2) and (rim_vert_length > rim_vert_tolerance_1)

        return (rim_horz_check and rim_vert_check)

    def _check_pod_inserted(self):
        """
        check if pod has been inserted successfully
        """
        pod_holder_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod_holder"]])
        pod_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["coffee_pod"]])

        # center of pod cannot be more than the difference of radii away from the center of pod holder
        pod_horz_check = True
        r_diff = self.pod_holder_size[0] - self.pod_size[0]
        pod_horz_check = (np.linalg.norm(pod_pos[:2] - pod_holder_pos[:2]) <= r_diff)

        # check that bottom of pod is within some tolerance of bottom of container
        pod_insertion_z_tolerance = 0.02
        z_lim_low = pod_holder_pos[2] - self.pod_holder_size[2]
        pod_z_check = (pod_pos[2] - self.pod_size[2] > z_lim_low) and (pod_pos[2] - self.pod_size[2] < z_lim_low + pod_insertion_z_tolerance)
        return (pod_horz_check and pod_z_check)

    def _check_lid_being_closed(self):
        """
        check if lid is being closed
        """

        # (check for hinge angle being less than default angle value, 120 degrees)
        hinge_angle = self.sim.data.qpos[self.hinge_qpos_addr]
        return (hinge_angle < 2.09)

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the coffee machine.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the coffee machine
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.coffee_machine)
