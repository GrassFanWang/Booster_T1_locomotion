import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from Booster_T1_locomotion.assets.robots.Booster_T1_Cfg import BOOSTER_T1_CFG
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import RewardTermCfg as RewTerm
import math
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.envs import ManagerBasedRLEnvCfg
from Booster_T1_locomotion.tasks.locomotion import mdp
from isaaclab.assets import Articulation

Terrain_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0,10.0),
    border_width=20.0,
    border_height=10.0,
    num_cols=10,
    num_rows=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5)
    }
)

Real_Joint_Name=[
            "Head_Yaw","Head_Pitch",                              
            "Left_Shoulder_Pitch","Left_Shoulder_Roll","Left_Elbow_Pitch","Left_Elbow_Yaw",
            "Right_Shoulder_Pitch","Right_Shoulder_Roll","Right_Elbow_Pitch","Right_Elbow_Yaw",
            "Waist",
            "Left_Hip_Pitch","Left_Hip_Roll","Left_Hip_Yaw","Left_Knee_Pitch",
            "Left_Ankle_Pitch","Left_Ankle_Roll",     
            "Right_Hip_Pitch","Right_Hip_Roll","Right_Hip_Yaw","Right_Knee_Pitch",
            "Right_Ankle_Pitch","Right_Ankle_Roll",
]

@configclass
class RobotSceneCfg(InteractiveSceneCfg):

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=Terrain_CFG,
        max_init_terrain_level=Terrain_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    

    randomize_rigid_body_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
           "asset_cfg": SceneEntityCfg("robot", body_names="^(?!.*TrunK).*"),  # 作用于除基座外的所有刚体（腿、关节等）
            "mass_distribution_params": (0.7, 1.3),  # 质量缩放范围：0.7~1.3倍原始质量
            "operation": "scale",  # 操作类型：缩放（按比例调整，更符合真实部件质量差异）
            "recompute_inertia": True,
        },
    )

    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,  # 随机化刚体质心位置
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),  # 所有刚体
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},  # 质心偏移范围：xyz各±0.05米
        },
    )
    

    # reset
    randomize_apply_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,  # 给刚体施加外力/扭矩
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),  # 作用于基座
            "force_range": (-10.0, 10.0),  # 外力范围：±10牛顿（N）
            "torque_range": (-10.0, 10.0), # 扭矩范围：±10牛·米（N·m）
        },
    )


    randomize_reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,  # 按缩放因子重置关节（也可改用reset_joints_by_offset）
        mode="reset",
        params={
            "position_range": (1.0, 1.0),  # 位置缩放因子：1.0（无随机，关节回到默认中立位）
            "velocity_range": (0.0, 0.0),  # 速度范围：0（关节初始无运动）
        },
    )


    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # interval
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)
        ),
    )



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=Real_Joint_Name, 
        scale=0.25, 
        use_default_offset=True, 
        clip={".*": (-100.0, 100.0)}, 
        preserve_order=True
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=2.0,
        )

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, 
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, 
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=Real_Joint_Name, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, 
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=Real_Joint_Name, preserve_order=True)},
            scale=0.05, 
            noise=Unoise(n_min=-1.5, n_max=1.5))
        
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        #     scale=1.0,
        # )       
        
       

        def __post_init__(self):
            self.history_length = 1
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=Real_Joint_Name, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=Real_Joint_Name, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 1.0),
        #     scale=1.0,
        # )


        def __post_init__(self):
            self.history_length = 1
            self.enable_corruption = False
            self.concatenate_terms = True
            

    # privileged observations
    critic: CriticCfg = CriticCfg()

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    is_terminated = RewTerm(func=mdp.is_terminated,weight=-200.0)

    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)
    
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, 
        weight=-1.0e-7, 
        params={"asset_cfg": 
                SceneEntityCfg("robot", 
                joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"]
            )
        }
    )

    joint_acc_l2_legs = RewTerm(
        func=mdp.joint_acc_l2, 
        weight=-1.25e-7, 
        params={"asset_cfg": 
                SceneEntityCfg("robot", 
                joint_names=[".*_Hip_.*", ".*_Knee_.*"])}
    )

    joint_acc_l2_arms = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.0e-7, # 给予适度惩罚
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Shoulder_.*", ".*_Elbow_.*"])}
    )

    # 增加对手臂速度的惩罚（防止高频乱晃）
    joint_vel_l2_arms = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Shoulder_.*", ".*_Elbow_.*"])}
    )
    
    joint_deviation_hip_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2,
        params={"asset_cfg": 
                SceneEntityCfg("robot", 
                joint_names=[".*_Hip_Yaw", ".*_Hip_Roll"]
            )
        }
    )

    joint_deviation_arms_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": 
                SceneEntityCfg("robot", 
                joint_names=[".*_Shoulder_.*", ".*_Elbow_.*"]
            )
        }
    )

    joint_deviation_torso_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": 
                SceneEntityCfg("robot", 
                joint_names="Waist"
            )
        }
    )

    joint_deviation_head_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": 
                SceneEntityCfg("robot", 
                joint_names="Head_.*"
            )
        }
    )

    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=Real_Joint_Name)}
    )

    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=Real_Joint_Name),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.075)
    
    

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=5.0,
        params={"command_name": 
                "base_velocity", 
                "std": math.sqrt(0.25)},
    )

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=3.5, 
        params={"command_name": 
                "base_velocity", 
                "std": math.sqrt(0.25)}
    )
    
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
        },
    )
    
    feet_distance = RewTerm(
        func=mdp.feet_distance_l2,
        weight=-0.5,
        params={
            "min_dist": 0.15,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot_link")
        },
    )

    # alive = RewTerm(func=mdp.is_alive, weight=0.15)
    # base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    # joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    # energy = RewTerm(func=mdp.energy, weight=-2e-5)
    
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_link"),
        },
    )
    
    upward = RewTerm(func=mdp.upward, weight=1.0)

    # joint_deviation_arms = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*Shoulder_Pitch",
    #                 ".*Shoulder_Roll",
    #                 ".*Elbow_Pitch",
    #                 ".*Elbow_Yaw",
    #             ],
    #         )
    #     },
    # )

    # joint_deviation_waists = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-1,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 "Waist",
    #             ],
    #         )
    #     },
    # )

    # joint_deviation_legs = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-1.0,
    #     params={"asset_cfg": 
    #             SceneEntityCfg("robot", 
    #             joint_names=[
    #                 ".*Hip_Roll",
    #                 ".*Hip_Yaw",
    #             ]
    #         )
    #     },
    # )


    #base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.65})
    
        # -- feet
    # gait = RewTerm(
    #     func=mdp.feet_gait,
    #     weight=0.5,
    #     params={
    #         "period": 0.8,
    #         "offset": [0.0, 0.5],
    #         "threshold": 0.55,
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_link"),
    #     },
    # )

    # feet_clearance = RewTerm(
    #     func=mdp.foot_clearance_reward,
    #     weight=1.0,
    #     params={
    #         "std": 0.05,
    #         "tanh_mult": 2.0,
    #         "target_height": 0.1,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*foot_link"),
    #     },
    # )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})
    # bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # command_resample
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )

    # Contact sensor
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Trunk"), "threshold": 1.0},
    )


# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

#     terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
#     lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)

@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5) # type: ignore
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg() # type: ignore
    actions: ActionsCfg = ActionsCfg() # type: ignore
    commands: CommandsCfg = CommandsCfg() # type: ignore
    # MDP settings
    rewards: RewardsCfg = RewardsCfg() # type: ignore
    terminations: TerminationsCfg = TerminationsCfg() # type: ignore
    events: EventCfg = EventCfg() # type: ignore
    # curriculum: CurriculumCfg = CurriculumCfg() # type: ignore

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
