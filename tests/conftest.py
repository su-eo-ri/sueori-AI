import importlib.util
import sys
from pathlib import Path

# tools/gen_vectors.py를 `tools_gen` 모듈로 불러올 수 있게 한다(tools/는 패키지가 아님).
_path = Path(__file__).resolve().parent.parent / "tools" / "gen_vectors.py"
_spec = importlib.util.spec_from_file_location("tools_gen", _path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["tools_gen"] = _mod
_spec.loader.exec_module(_mod)
