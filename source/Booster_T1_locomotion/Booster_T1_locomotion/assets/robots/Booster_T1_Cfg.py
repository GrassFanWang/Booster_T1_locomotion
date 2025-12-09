import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

@configclass
class BoosterArticulationCfg(ArticulationCfg):

    joint_sdk_names: list[str] = None
    soft_joint_pos_limit_factor = 0.9

@configclass
class BoosterUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )

BOOSTER_T1_CFG = BoosterArticulationCfg(

    spawn=BoosterUsdFileCfg(
        usd_path="/home/grassfan-wang/Projects/Booster_T1_locomotion/T1_usd/t1/t1.usd"
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.68),
        joint_pos={
                   "Left_Shoulder_Roll": -1.57,
                   "Right_Shoulder_Roll": 1.57,
                   },
        joint_vel={".*": 0.0},   
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*Hip_Roll",
                ".*Hip_Yaw",
                ".*Hip_Pitch",
                ".*Knee_Pitch",
            ],
            effort_limit_sim={
                ".*_Hip_Pitch": 45.0,
                ".*_Hip_Roll": 30.0,
                ".*_Hip_Yaw": 30.0,
                ".*_Knee_Pitch": 60.0,
            },
            velocity_limit_sim={
                ".*_Hip_Pitch": 12.5,
                ".*_Hip_Roll": 10.9,
                ".*_Hip_Yaw": 10.9,
                ".*_Knee_Pitch": 11.7,
            },
            damping={
            ".*(Hip_(Roll|Yaw|Pitch)|Knee_Pitch)": 5.0,
            },
            stiffness=200.0,
            armature=0.01,
        
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*Ankle_Pitch",
                ".*Ankle_Roll",
            ],
            effort_limit_sim={
                ".*_Ankle_Pitch": 24, 
                ".*_Ankle_Roll": 15
            },
            velocity_limit_sim={
                ".*_Ankle_Pitch": 18.8, 
                ".*_Ankle_Roll": 12.4
            },
            stiffness=50.0,
            damping=1.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*Shoulder_Pitch",
                ".*Shoulder_Roll",
                ".*Elbow_Pitch",
                ".*Elbow_Yaw",
            ],
            effort_limit_sim=18.0,
            velocity_limit_sim=18.8,
            stiffness=40.0,
            damping=10.0,
            armature=0.01,
        ),
        "waist":ImplicitActuatorCfg(
            joint_names_expr=[
               "Waist",
            ],
            effort_limit_sim=30.0,
            velocity_limit_sim=10.88,
            stiffness=200.0,
            damping=5.0,
            armature=0.01,
        ),
    },
    joint_sdk_names=[
        "Left_Shoulder_Pitch",
        "Right_Shoulder_Pitch",
        "Waist",
        "Head_Yaw",
        "Head_Pitch",
        "Left_Shoulder_Roll",
        "Left_Elbow_Pitch",
        "Left_Elbow_Yaw",
        "Right_Shoulder_Roll",
        "Right_Elbow_Pitch",
        "Right_Elbow_Yaw",
        "Left_Hip_Pitch",
        "Right_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Knee_Pitch",
        "Left_Ankle_Pitch",
        "Left_Ankle_Roll",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Knee_Pitch",
        "Right_Ankle_Pitch",
        "Right_Ankle_Roll",
    ]

)    
