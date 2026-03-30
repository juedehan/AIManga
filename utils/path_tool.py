import importlib.util
import os
from pathlib import Path

_BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "bootstrap.py"
_BOOTSTRAP_SPEC = importlib.util.spec_from_file_location("aimanga_bootstrap", _BOOTSTRAP_PATH)
if _BOOTSTRAP_SPEC is None or _BOOTSTRAP_SPEC.loader is None:
    raise ImportError(f"无法加载 bootstrap: {_BOOTSTRAP_PATH}")
_bootstrap = importlib.util.module_from_spec(_BOOTSTRAP_SPEC)
_BOOTSTRAP_SPEC.loader.exec_module(_bootstrap)
_bootstrap.ensure_project_root()

"""
为整个工程提供统一的绝对路径
"""
import os

def get_project_root() -> str:
    """
    获取工程所在的根目录
    :return: 字符串根目录
    """
    # 当前文件的绝对路径
    current_file= os.path.abspath(__file__)
    # 获取工程的根目录，先获取文件所在的文件夹的绝对路径
    current_dir=os.path.dirname(current_file)
    # 获取工程根目录
    project_root=os.path.dirname(current_dir)

    return project_root

def get_abs_path(relative_path:str)->str:
    """
    传递相对路径，得到绝对路径
    :param relative_path:
    :return:
    """
    project_root=get_project_root()
    return os.path.join(project_root,relative_path)

if __name__ == '__main__':
    print(get_abs_path("config/config.txt"))
