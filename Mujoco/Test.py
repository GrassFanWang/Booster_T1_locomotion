import mujoco
import mujoco.viewer
import time
import numpy as np
from pathlib import Path
from collections import deque
import torch


        # "Head_Yaw",              
        # "Head_Pitch",           
        # "Left_Shoulder_Pitch",   
        # "Left_Shoulder_Roll",    
        # "Left_Elbow_Yaw",        
        # "Left_Elbow_Pitch",      
        # "Right_Shoulder_Pitch",  
        # "Right_Shoulder_Roll",  
        # "Right_Elbow_Yaw",       
        # "Right_Elbow_Pitch",    
        # "Waist",                 
        # "Left_Hip_Pitch",       
        # "Left_Hip_Roll",         
        # "Left_Hip_Yaw",          
        # "Left_Knee_Pitch",       
        # "Left_Ankle_Pitch",      
        # "Left_Ankle_Roll",       
        # "Right_Hip_Pitch",      
        # "Right_Hip_Roll",       
        # "Right_Hip_Yaw",        
        # "Right_Knee_Pitch",      
        # "Right_Ankle_Pitch",    
        # "Right_Ankle_Roll",      Right_Ankle_Roll        22

class T1RobotSim:
    
    def __init__(self):
        
        
        self.joint_names = [
            "Head_Yaw", "Head_Pitch", 
            "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Yaw", "Left_Elbow_Pitch",
            "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Yaw", "Right_Elbow_Pitch",
            "Waist",
            "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
            "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll"
        ]
        
        # 既然顺序一致了，就不再需要 self.isaac_to_mujoco_idx 映射表
        self.default_joint_pos = np.zeros(23)
        self.device = torch.device("cpu")
        self.load_model()
        self.reset_robot_pose()
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Trunk")
        
      
        self.action_netput = np.zeros(23)
        self.action = np.zeros(23)
        self.last_action = np.zeros(23)
        self.command = np.zeros(3)
      
        self.scales = {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "gravity": 1.0,
            "cmd": 1.0,
            "joint_pos": 1.0,
            "joint_vel": 0.05,
            "action": 1.0,
            "height": 1.0
        }
        self.joint_damping = np.zeros(23)
        self.joint_stiffness = np.zeros(23)
        self.torque = np.zeros(23)

        for i in range(23):
            if i <= 1:
                self.joint_stiffness[i] = 50.0
                self.joint_damping[i] = 1.0
            if  1 < i and i <= 9:
                self.joint_stiffness[i] = 40.0
                self.joint_damping[i] = 10.0

            if i == 10:
                self.joint_stiffness[i] = 200.0
                self.joint_damping[i] = 5.0
            
            if 10<i and i<=14:
                self.joint_stiffness[i] = 200.0
                self.joint_damping[i] = 5.0
            
            if 14<i and i<=16:
                self.joint_stiffness[i] = 50.0
                self.joint_damping[i] = 1.0
            
            if 16<i and i<=20:
                self.joint_stiffness[i] = 200.0
                self.joint_damping[i] = 5.0
                
            if 20<i and i<=22:
                self.joint_stiffness[i] = 50.0
                self.joint_damping[i] = 1.0

        
        self.obs_history_len = 1
        self.obs_dim_single = 81 # 单帧维度
        # 初始化队列，自动把旧数据挤出去
        self.obs_history = deque(maxlen=self.obs_history_len)
        # 先填充零数据
        for _ in range(self.obs_history_len):
            self.obs_history.append(np.zeros(self.obs_dim_single))

        self.load_policy(self.policy_path)

    def load_policy(self, path):
        print(f"Loading JIT policy from: {path}")
        self.policy = torch.jit.load(path, map_location=self.device)
        self.policy.eval()



    def get_project_root(self):
        
        current = Path(__file__).resolve().parent
        for parent in [current] + list(current.parents):
            if (parent / '.git').exists():
                return parent
        return current
    
    def load_model(self):
         
        ROOT_DIR = self.get_project_root()
        MODEL_PATH = (ROOT_DIR / "T1_usd" / "T1_23dof.xml").as_posix() 
        self.policy_path = (ROOT_DIR / "logs" / "rsl_rl" / "2025-12-20_15-15-38" / "exported" / "policy.pt")
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        print(f"成功加载模型: {MODEL_PATH}")



    def get_actuator_index(self, actuator_name):
        """获取执行器索引"""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
         

    def reset_robot_pose(self):

        initial_qpos = {
                "Left_Shoulder_Roll": -1.35, "Right_Shoulder_Roll": 1.35,
                "Left_Shoulder_Pitch": 0.2, "Right_Shoulder_Pitch": 0.2,
                "Left_Elbow_Yaw": -0.5, "Right_Elbow_Yaw": 0.5,
                "Left_Hip_Pitch": -0.20, "Right_Hip_Pitch": -0.20,
                "Left_Knee_Pitch": 0.42, "Right_Knee_Pitch": 0.42,
                "Left_Ankle_Pitch": -0.23, "Right_Ankle_Pitch": -0.23
        }

        # 1. 重置所有关节为0
        self.data.qpos[7:] = 0.0
        
        # 2. 应用初始姿态 并 记录到 default_dof_pos
        for joint_name, angle in initial_qpos.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id != -1:
                # 获取该关节在 qpos 数组中的索引 (偏移量 7 是因为前7个是自由关节)
                # 更稳健的方法是使用 qposadr，但这里假设顺序一致，我们手动填充 default_dof_pos
                # 注意：self.joint_names 的顺序必须与 mujoco qpos 顺序一致
                try:
                    idx = self.joint_names.index(joint_name)
                    self.default_joint_pos[idx] = angle # <--- 记录默认值
                    
                    # 同时设置 MuJoCo 的状态
                    qpos_adr = self.model.jnt_qposadr[joint_id]
                    self.data.qpos[qpos_adr] = angle
                except ValueError:
                    pass
            else:
                print(f"警告: 找不到关节 {joint_name}")
        
        for joint_name, angle in initial_qpos.items():
                try:
                    self.data.joint(joint_name).qpos = angle
                except Exception as e:
                    print(f"警告: 无法设置关节 {joint_name} - {e}")
            
        print(self.data.qpos)
        mujoco.mj_forward(self.model, self.data)
     
    def get_rotation_matrix(self):
        """获取基座到世界的旋转矩阵 R_world_base"""
        # 假设 body 1 是躯干 (需根据你的 XML 确认 id)
        # data.xmat 是 flattened 3x3 array
        # 使用 sensor 数据更稳健，或者直接取 body xmat
        return self.data.xmat[1].reshape(3, 3)
    
    def compute_obs(self,command,action_net_output):
        
        
        qpos = self.data.qpos
        qvel = self.data.qvel
        R = self.get_rotation_matrix()
        R_inv = R.T
        
        v_world = qvel[0:3]
        base_lin_vel = R_inv @ v_world * self.scales["lin_vel"] 
        base_ang_vel = self.data.sensor("angular-velocity").data * self.scales["ang_vel"] 
        
        gravity_vec = np.array([0, 0, -1])
        projected_gravity = R_inv @ gravity_vec * self.scales["gravity"] 
         
        velocity_commands = command * self.scales["cmd"] 

        # --- 关键简化部分 ---
        # 既然顺序一致，直接切片即可 (MuJoCo qpos 偏移 7, qvel 偏移 6)
        current_joint_pos = qpos[7:]  
        current_joint_vel = qvel[6:]

        # 直接计算，无需 loop 映射
        # (关节位置 - 默认位置) * 缩放
        joint_pos_rel = (current_joint_pos - self.default_joint_pos) * self.scales["joint_pos"]
        joint_vel_rel = current_joint_vel * self.scales["joint_vel"]
        
        self.action = action_net_output.copy()

        obs_list = [
            base_lin_vel,       # 3
            base_ang_vel,       # 3
            projected_gravity,  # 3
            velocity_commands,  # 3
            joint_pos_rel,      # 23
            joint_vel_rel,      # 23  
            -self.action,        # 23
        ] # 合计: 3+3+3+3+23+23+23 = 81 维

        flat_obs = np.concatenate([x.flatten() for x in obs_list])
        return np.clip(flat_obs, -100.0, 100.0)

    def get_stacked_obs(self, current_obs_frame):
        """将当前帧加入历史队列，并返回堆叠后的观测"""
        self.obs_history.append(current_obs_frame)
        # 拼接 deque 中的所有帧
        stacked_obs = np.concatenate(list(self.obs_history))
        return stacked_obs # 形状应该是 (1455,)
    
    def pd_control(self,action_from_policy):
        """执行 PD 控制 (顺序已对齐)"""
        
        # 1. 计算目标位置：Target = Default + Action * Action_Scale
        # 注意：这里的 0.5 应对应你训练时的 action_scale
        target_pos = self.default_joint_pos + action_from_policy * 0.2
        
        current_pos = self.data.qpos[7:]
        current_vel = self.data.qvel[6:]

        # 2. 计算 PD 力矩
        # 这里的 joint_stiffness 数组顺序必须和 joint_names 一致
        torques = self.joint_stiffness * (target_pos - current_pos) - self.joint_damping * current_vel
        
        # 3. 限制力矩并应用
        torques = np.clip(torques, -80, 80)
        
        # 将计算好的 23 个力矩一次性赋值给 MuJoCo 控制量
        # 假设你的 XML 中 actuator 的顺序也是按这 23 个关节排列的
        self.data.ctrl[:] = torques

    def step(self):

        # 这里是物理步进的核心
        mujoco.mj_step(self.model, self.data)





        
    def run(self):
            print("开始仿真...")
            start = time.time()
            simulation_duration = 60
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # 这里的 decimation (10) 应与训练配置 self.decimation * (训练步长/物理步长) 匹配
                
                while viewer.is_running() and time.time() - start < simulation_duration:
                    step_start = time.time()

                    # 1. 获取观测 (此时 self.action_netput 是上一帧的结果)
                    obs_frame = self.compute_obs(self.command, self.action_netput)
                    full_obs = self.get_stacked_obs(obs_frame)    
                    
                    # 2. 推理新动作
                    with torch.no_grad():
                        obs_tensor = torch.from_numpy(full_obs).float().unsqueeze(0).to(self.device)
                        self.action_netput = self.policy(obs_tensor).squeeze().cpu().numpy()
                    #self.action_netput = np.zeros(23)
                    # 3. 物理子步循环 (Decimation)
                    for _ in range(decimation):
                        self.pd_control(self.action_netput) # 应用对齐后的新动作
                        self.step() 

                    viewer.sync()
                
                    # 时间补偿保持实时
                    dt = self.model.opt.timestep * decimation
                    time_until_next_step = dt - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)

class Observer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
    
    def compute_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        print(qvel)


def main():
        
        sim = T1RobotSim()
        
        sim.run()

if __name__ == "__main__":
    main()

    







# with mujoco.viewer.launch_passive(model, model_data) as viewer:
#   # Close the viewer automatically after 30 wall-seconds.
#   start = time.time()
#   while viewer.is_running() and time.time() - start < 30:
#     step_start = time.time()

#     # mj_step can be replaced with code that also evaluates
#     # a policy and applies a control signal before stepping the physics.
#     mujoco.mj_step(model, model_data)

#     model_data.ctrl[0] = 1

#     # Pick up changes to the physics state, apply perturbations, update options from GUI.
#     viewer.sync()

#     # Rudimentary time keeping, will drift relative to wall clock.
#     time_until_next_step = model.opt.timestep - (time.time() - step_start)
#     if time_until_next_step > 0:
#       time.sleep(time_until_next_step)