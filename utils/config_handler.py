import importlib.util
from pathlib import Path

_BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "bootstrap.py"
_BOOTSTRAP_SPEC = importlib.util.spec_from_file_location("aimanga_bootstrap", _BOOTSTRAP_PATH)
if _BOOTSTRAP_SPEC is None or _BOOTSTRAP_SPEC.loader is None:
    raise ImportError(f"无法加载 bootstrap: {_BOOTSTRAP_PATH}")
_bootstrap = importlib.util.module_from_spec(_BOOTSTRAP_SPEC)
_BOOTSTRAP_SPEC.loader.exec_module(_bootstrap)
_bootstrap.ensure_project_root()

"""
yml
k : V
"""
import yaml
from utils.path_tool import get_abs_path


def _load_yaml_config(config_path: str, encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_rag_config(config_path: str = get_abs_path("config/rag.yml"), encoding: str = "utf-8"):
    return _load_yaml_config(config_path, encoding)


def load_model_config(config_path: str = get_abs_path("config/model.yml"), encoding: str = "utf-8"):
    return _load_yaml_config(config_path, encoding)


def load_chroma_config(config_path: str = get_abs_path("config/chroma.yml"), encoding: str = "utf-8"):
    return _load_yaml_config(config_path, encoding)


def load_rag_service_config(config_path: str = get_abs_path("config/rag_service.yml"), encoding: str = "utf-8"):
    return _load_yaml_config(config_path, encoding)


def load_vector_store_config(config_path: str = get_abs_path("config/vector_store.yml"), encoding: str = "utf-8"):
    return _load_yaml_config(config_path, encoding)


def load_prompts_config(config_path: str = get_abs_path("config/prompts.yml"), encoding: str = "utf-8"):
    return _load_yaml_config(config_path, encoding)


def load_agent_config(config_path: str = get_abs_path("config/agent.yml"), encoding: str = "utf-8"):
    return _load_yaml_config(config_path, encoding)


def load_data_config(config_path: str = get_abs_path("config/data.yml"), encoding: str = "utf-8"):
    return _load_yaml_config(config_path, encoding)


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
rag_service_conf = load_rag_service_config()
vector_store_conf = load_vector_store_config()
agent_conf = load_agent_config()
prompts_conf = load_prompts_config()
model_conf = load_model_config()
data_conf = load_data_config()

if __name__ == '__main__':
    print(rag_conf)
