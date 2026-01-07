import torch 
from pathlib import Path

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

model_path = (ROOT_DIR / "logs" / "rsl_rl" / "2025-12-20_15-15-38" / "exported" / "policy.pt").as_posix()

print(f"Loading JIT model from: {model_path}")
# 注意：这里必须用 torch.jit.load，而不是 torch.load
model = torch.jit.load(model_path)

# 获取包含所有参数（权重和偏置）的字典
state_dict = model.state_dict()

for key, value in state_dict.items():
    print(f"Layer: {key} | Shape: {value.shape}")