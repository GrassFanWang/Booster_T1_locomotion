from omni.isaac.core.articulations import Articulation

# 替换为你的机器人路径
prim_path = "/World/T1" 
art = Articulation(prim_path)
art.initialize()

# 打印所有关节名称（按索引顺序）
print("Joint Names (Order):", art.dof_names)
print("Number of DOFs:", art.num_dof)