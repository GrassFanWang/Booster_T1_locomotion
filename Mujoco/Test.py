import mujoco
import mujoco.viewer
import time
import numpy as np
from pathlib import Path
from collections import deque
import torch
import sys
import glfw # 确保开头导入了 glfw
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
        # "Right_Ankle_Roll",    
        # 
        #        

class T1RobotSim:
    
    def __init__(self):
        
        self.load_model()
        
        self.initial_qpos = {
                "Left_Shoulder_Roll": -1.35, "Right_Shoulder_Roll": 1.35,
                "Left_Shoulder_Pitch": 0.2, "Right_Shoulder_Pitch": 0.2,
                "Left_Elbow_Yaw": -0.5, "Right_Elbow_Yaw": 0.5,
                "Left_Hip_Pitch": -0.20, "Right_Hip_Pitch": -0.20,
                "Left_Knee_Pitch": 0.42, "Right_Knee_Pitch": 0.42,
                "Left_Ankle_Pitch": -0.23, "Right_Ankle_Pitch": -0.23
        }

        self.isaac_joint_names = [
        'Head_Yaw', 'Left_Shoulder_Pitch', 'Right_Shoulder_Pitch', 'Waist', 'Head_Pitch', 
        'Left_Shoulder_Roll', 'Right_Shoulder_Roll', 'Left_Hip_Pitch', 'Right_Hip_Pitch', 
        'Left_Elbow_Pitch', 'Right_Elbow_Pitch', 'Left_Hip_Roll', 'Right_Hip_Roll', 
        'Left_Elbow_Yaw', 'Right_Elbow_Yaw', 'Left_Hip_Yaw', 'Right_Hip_Yaw', 
        'Left_Knee_Pitch', 'Right_Knee_Pitch', 'Left_Ankle_Pitch', 'Right_Ankle_Pitch', 
        'Left_Ankle_Roll', 'Right_Ankle_Roll'
       ]
         # 计算映射逻辑
        # 我们问：Isaac 里的第 j 个关节，在 MuJoCo 列表里排第几？
            # 2. 动态获取 MuJoCo 当前模型中的关节名称 (排除 world_joint)
        # 注意：jnt_qposadr 决定了 qpos 里的实际存储位置
        self.mujoco_joint_names = []
        for i in range(self.model.njnt):
            name = self.model.joint(i).name
            if name != "world_joint": # 假设自由根关节叫 world_joint 或 root
                 self.mujoco_joint_names.append(name)
        

        # 3. 计算映射索引
        # mu_to_is_indices: 告诉程序，Isaac 顺序中的每个关节在 MuJoCo 数组里的哪个位置
        # 示例: isaac_joint_names[0] (Head_Yaw) 在 mujoco_joint_names 里的索引是几？
        self.mu_to_is_indices = np.array([ self.mujoco_joint_names.index(name) for name in self.isaac_joint_names])

        # is_to_mu_indices: 告诉程序，MuJoCo 顺序中的每个关节在 Isaac 数组里的哪个位置
        # 用于将 Policy 输出 (Isaac 顺序) 转回 MuJoCo (ctrl 顺序)
        self.is_to_mu_indices = np.array([self.isaac_joint_names.index(name) for name in  self.mujoco_joint_names])

        # 4. 对齐默认姿态 (重要！)
        # 确保你的 default_joint_pos 也是按照 Isaac 顺序排列的
        self.default_joint_pos_is = np.zeros(len(self.isaac_joint_names), dtype=np.float32)
        # ... 在这里根据名称填充 default_joint_pos_is ...

        self.joint_stiffness = [   
            4.0, 4.0,
            4.0, 4.0, 4.0, 4.0,
            4.0, 4.0, 4.0, 4.0,
            50.0,
            200., 200., 200., 200., 200., 200.,
            200., 200., 200., 200., 50., 50.,
        ]

        self.joint_damping = [   
            1., 1.,
            1., 1., 1., 1.,
            1., 1., 1., 1.,
            1.,
            5., 5., 5., 5., 5., 5.,
            5., 5., 5., 5., 1, 1,
        ]
        
        self.default_joint_pos = np.zeros(23,dtype=np.float32)
        self.target_joint_pos = np.zeros(23,dtype=np.float32)
        self.device = torch.device("cpu")
        self.reset_robot_pose()
        self.action = np.zeros(23,dtype=np.float32)
        self.command = np.zeros(3)
        self.scales = {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "gravity": 1.0,
            "cmd": 1.0,
            "joint_pos": 1.0,
            "joint_vel": 0.05,
            "action": 1.0, 
        }
        self.torque = np.zeros(23,dtype=np.float32)
        self.load_policy(self.policy_path)
        self.obs = np.zeros(81, dtype=np.float32)
        
        # 假设 ground 的 geom id 是 0
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

        # 修改滑动摩擦力为 0.1 (非常滑)
        self.model.geom_friction[geom_id][0] = 1.0
    
    def get_project_root(self):
        current = Path(__file__).resolve().parent
        for parent in [current] + list(current.parents):
            if (parent / '.git').exists():
                return parent
        return current

    

    def load_policy(self, path):
        print(f"Loading JIT policy from: {path}")
        self.policy = torch.jit.load(path)

    
    def load_model(self):
         
        ROOT_DIR = self.get_project_root()
        MODEL_PATH = (ROOT_DIR / "T1_usd" / "T1_23dof.xml").as_posix() 
        self.policy_path = (ROOT_DIR / "logs" / "rsl_rl" / "2025-12-20_15-15-38" / "exported" / "policy.pt")
        
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        
        self.data = mujoco.MjData(self.model)
        #self.model.opt.gravity[:] = [0, 0, 0]
        
        print(f"成功加载模型: {MODEL_PATH}")

    

    def get_actuator_index(self, actuator_name):
        
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
         

    def reset_robot_pose(self):

        

        # 1. 重置所有关节为0
        self.data.qpos[7:] = 0.0
        
        # 2. 应用初始姿态 并 记录到 default_dof_pos
        for joint_name, angle in self.initial_qpos.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id != -1:
                # 获取该关节在 qpos 数组中的索引 (偏移量 7 是因为前7个是自由关节)
                # 更稳健的方法是使用 qposadr，但这里假设顺序一致，我们手动填充 default_dof_pos
                # 注意：self.joint_names 的顺序必须与 mujoco qpos 顺序一致
                try:
                    idx =  self.mujoco_joint_names.index(joint_name)
                    self.default_joint_pos[idx] = angle # <--- 记录默认值
                    
                    # 同时设置 MuJoCo 的状态
                    qpos_adr = self.model.jnt_qposadr[joint_id]
                    self.data.qpos[qpos_adr] = angle
                except ValueError:
                    pass
            else:
                print(f"警告: 找不到关节 {joint_name}")
        
        for joint_name, angle in self.initial_qpos.items():
                try:
                    self.data.joint(joint_name).qpos = angle
                except Exception as e:
                    print(f"警告: 无法设置关节 {joint_name} - {e}")
        
        self.default_joint_pos_is = self.default_joint_pos[self.mu_to_is_indices]
        #self.target_joint_pos = self.default_joint_pos.copy()
        
        print(self.default_joint_pos)
        print(self.default_joint_pos_is)

    def get_gravity_orientation(self,quat):
        qw = quat[0]
        qx = quat[1]
        qy = quat[2]
        qz = quat[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation
    
    def compute_obs(self,command):
        
        q_mu = self.data.qpos[7:].astype(np.float32)
        dq_mu = self.data.qvel[6:].astype(np.float32)
        
        q_is = q_mu[self.mu_to_is_indices]
        dq_is = dq_mu[self.mu_to_is_indices]
        # print(self.default_joint_pos_is)
        # print(q_is)


        world_lin_vel = self.data.qvel[:3].astype(np.float32)

        # 2. 获取机器人基座（通常是第一个 body）的旋转矩阵
        # self.data.xmat[1] 获取 ID 为 1 的 body 的旋转矩阵 (通常 0 是 worldbody)
        # 如果你不确定 ID，可以用 self.data.body('trunk').xmat
        # xmat 是 9个元素的数组，需要 reshape 成 3x3
        trunk_id = 1 # 或者是 self.model.body('trunk').id
        rot_matrix = self.data.xmat[trunk_id].reshape(3, 3)

        # 3. 将世界系速度投影到机体系：v_body = R.T @ v_world
        base_lin_vel = rot_matrix.T @ world_lin_vel
        base_ang_vel = self.data.sensor("angular-velocity").data.astype(np.float32) * self.scales["ang_vel"] 
        
        quat = self.data.sensor("orientation").data.astype(np.float32)
        #gravity_vec = np.array([0, 0, -1])
        projected_gravity = self.get_gravity_orientation(quat) * self.scales["gravity"] 
           
        velocity_commands = command * self.scales["cmd"] 

        # --- 关键简化部分 ---
        # 既然顺序一致，直接切片即可 (MuJoCo qpos 偏移 7, qvel 偏移 6)
        current_joint_pos = q_is
        current_joint_vel = dq_is

        joint_pos_rel = (current_joint_pos - self.default_joint_pos_is) * self.scales["joint_pos"]
        joint_vel_rel = current_joint_vel * self.scales["joint_vel"]
        
        self.obs[0:3] = base_lin_vel  # 0-2
        self.obs[3:6] = base_ang_vel # 3-5
        self.obs[6:9] = projected_gravity #6-8
        self.obs[9:12] = velocity_commands #9-11
        self.obs[12:35] = joint_pos_rel #12-34
        self.obs[35:58] = joint_vel_rel #35-57
        self.obs[58:81] = self.action #58-80
        
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        
        action_mu = self.action[self.is_to_mu_indices]
        
        self.target_joint_pos =  self.default_joint_pos + action_mu * 0.5
    

    
    def pd_control(self):
        q_mu = self.data.qpos[7:].astype(np.float32)
        dq_mu = self.data.qvel[6:].astype(np.float32)



        # 计算 Isaac 顺序下的力矩
        # 确保 stiffness 和 damping 也是 Isaac 顺序，或者在这里进行对应处理
        # 建议在 __init__ 里把 stiffness 改成 Isaac 顺序
        torques = self.joint_stiffness * (self.target_joint_pos - q_mu) - self.joint_damping * dq_mu


        self.data.ctrl[:] = np.clip(torques, -100, 100)
  
    def step(self):

        # 这里是物理步进的核心
        mujoco.mj_step(self.model, self.data)



    def run(self):
            print("开始仿真...")
            counter = 0
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # 这里的 decimation (10) 应与训练配置 self.decimation * (训练步长/物理步长) 匹配
                decimation = 5
                track_body_id = 1 # 默认通常 1 是机器人的第一个 body

                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = track_body_id
                while viewer.is_running():
                    step_start = time.time()
                    self.pd_control()
                    self.step()
                    counter+=1
                    
                    
                    if counter % decimation == 0:
                    
                         
                        self.command = np.array([0.5,0,0], dtype=np.float32)
                    
            
                        self.compute_obs(self.command)
                        sys.stdout.write("\033[H\033[J") 
                        sys.stdout.flush()

                        print("========= T1 Robot Observation Monitor =========")
                        print(f"Time: {self.data.time:.2f}s | Counter: {counter}")
                        print(f"base_lin_vel: {self.obs[0:3]}")
                        print(f"base_ang_vel:  {self.obs[3:6]}")
                        print(f"projected_gravity:  {self.obs[6:9]}")
                        print(f"velocity_commands:  {self.obs[9:12]}")
                        print(f"joint_pos_rel:  {self.obs[12:35]}")
                        print(f"joint_vel_rel:  {self.obs[35:58]}")
                        print(f"action:  {self.obs[58:81]}")
                        print("================================================")

                    viewer.sync()
                
                    # 时间补偿保持实时
                    time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)




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
#       time.sleep(time_until_next_step)# 假设 ground 的 geom id 是 0
