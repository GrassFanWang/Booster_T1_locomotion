import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass
from pathlib import Path

ARMATURE_6416 = 0.095625
ARMATURE_4310 = 0.0282528
ARMATURE_6408 = 0.0478125
ARMATURE_4315 = 0.0339552
ARMATURE_8112 = 0.0523908
ARMATURE_8116 = 0.0636012
ARMATURE_ROB_14 = 0.001

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_6416 = 80.
STIFFNESS_4310 = 80.
STIFFNESS_6408 = 80.
STIFFNESS_4315 = 80.
STIFFNESS_ROB_14 = 4.

DAMPING_6416 = 2.0
DAMPING_4310 = 2.0
DAMPING_6408 = 2.0
DAMPING_4315 = 2.0
DAMPING_ROB_14 = 1.

def get_project_root():
    """
    向上递归查找，直到找到包含 .git 或 requirements.txt 的文件夹作为根目录
    """
    # 从当前文件所在目录开始
    current = Path(__file__).resolve().parent
    
    # 循环向上查找，直到根目录
    for parent in [current] + list(current.parents):
        # 检查特征文件是否存在
        if (parent / '.git').exists():
            return parent
    
    # 如果找不到，默认返回当前脚本的上一级
    return current

# 使用
ROOT_DIR = get_project_root()

# 以后不管你的脚本被移动到哪里，只要还
# 在项目里，这个路径都是对的
model_path = (ROOT_DIR / "T1_usd" / "t1" / "T1.usd").as_posix()

@configclass
class BoosterArticulationCfg(ArticulationCfg):
    
    joint_sdk_names: list[str] = None
    soft_joint_pos_limit_factor = 0.9

@configclass
class BoosterUsdFileCfg(sim_utils.UsdFileCfg):
    fix_base=False,
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
        enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )


BOOSTER_T1_CFG = BoosterArticulationCfg(
    spawn=BoosterUsdFileCfg(
        usd_path=model_path
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.68),
        joint_pos={
            "Head_Yaw" : 0.0,
            "Head_Pitch" : 0.0,
            # Arm
            ".*_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.35,
            "Right_Shoulder_Roll": 1.35,
            ".*_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
            # Waist
            "Waist": 0.0,
            # Leg
            ".*_Hip_Pitch": -0.20,
            ".*_Hip_Roll": 0.0,
            ".*_Hip_Yaw": 0.0,
            ".*_Knee_Pitch": 0.42,
            ".*_Ankle_Pitch": -0.23,
            ".*_Ankle_Roll": 0.0, 
            },
        joint_vel={".*": 0.0},   
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit_sim={
                ".*_Shoulder_Pitch": 18.0,
                ".*_Shoulder_Roll": 18.0,
                ".*_Elbow_Pitch": 18.0,
                ".*_Elbow_Yaw": 18.0,
            },
            velocity_limit_sim={
                ".*_Shoulder_Pitch": 7.33,
                ".*_Shoulder_Roll": 7.33,
                ".*_Elbow_Pitch": 7.33,
                ".*_Elbow_Yaw": 7.33,
            },
            stiffness={
                ".*_Shoulder_Pitch": 50.,
                ".*_Shoulder_Roll": 50.,
                ".*_Elbow_Pitch": 50.,
                ".*_Elbow_Yaw": 50.,
            },
            damping={
                ".*_Shoulder_Pitch": 1.,
                ".*_Shoulder_Roll": 1.,
                ".*_Elbow_Pitch": 1.,
                ".*_Elbow_Yaw": 1.,
            },
            armature={
                ".*_Shoulder_Pitch": ARMATURE_4310,
                ".*_Shoulder_Roll": ARMATURE_4310,
                ".*_Elbow_Pitch": ARMATURE_4310,
                ".*_Elbow_Yaw": ARMATURE_4310,
            },
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["Waist"],
            effort_limit_sim=25.0,
            velocity_limit_sim=12.57,
            stiffness=200.,
            damping=5.,
            armature=ARMATURE_6408,
        ),
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
            stiffness={
                ".*_Hip_Pitch": 200.,
                ".*_Hip_Roll": 200.,
                ".*_Hip_Yaw": 200.,
                ".*_Knee_Pitch": 200.,
            },
            damping={
                ".*_Hip_Pitch": 5.,
                ".*_Hip_Roll": 5.,
                ".*_Hip_Yaw": 5.,
                ".*_Knee_Pitch": 5.,
            },
            armature={
                ".*_Hip_Pitch": ARMATURE_8112,
                ".*_Hip_Roll": ARMATURE_6408,
                ".*_Hip_Yaw": ARMATURE_6408,
                ".*_Knee_Pitch": ARMATURE_8116,
            },
        
        ),
      
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*Ankle_Pitch",
                ".*Ankle_Roll",
            ],
            effort_limit_sim={
                ".*_Ankle_Pitch": 24.0, 
                ".*_Ankle_Roll": 15.0,
            },
            velocity_limit_sim={
                ".*_Ankle_Pitch": 18.8, 
                ".*_Ankle_Roll": 12.4
            },
            stiffness=50.0,
            damping=1.0,
            armature=2 * ARMATURE_4315,
        ),
        
        "head": ImplicitActuatorCfg(
            joint_names_expr=[
               "Head_Yaw" ,
               "Head_Pitch",
            ],
            effort_limit_sim=7.0,
            velocity_limit_sim=20.0,
            stiffness=10.0,
            damping=1.0,
            armature=0.001,
        ),
    },
    joint_sdk_names=[
        "Head_Yaw",
        "Head_Pitch",                              
        "Left_Shoulder_Pitch",
        "Left_Shoulder_Roll",
        "Left_Elbow_Pitch",
        "Left_Elbow_Yaw",
        "Right_Shoulder_Pitch",       
        "Right_Shoulder_Roll",
        "Right_Elbow_Pitch",
        "Right_Elbow_Yaw",
        "Waist",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Knee_Pitch",
        "Left_Ankle_Pitch",
        "Left_Ankle_Roll",     
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Knee_Pitch",
        "Right_Ankle_Pitch",
        "Right_Ankle_Roll",
    ]
  
)    
