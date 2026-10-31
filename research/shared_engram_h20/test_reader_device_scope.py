"""Device-map selection and restoration without a CUDA or NIXL dependency."""
import ast
import contextlib
import os
from pathlib import Path

source=Path(__file__).parent/'shared_model_adapter.py'
module=ast.parse(source.read_text())
node=next(x for x in module.body if isinstance(x,ast.FunctionDef) and x.name=='_reader_device_scope')
code=compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec')
namespace={'contextmanager':contextlib.contextmanager,'os':os};exec(code,namespace)
scope=namespace['_reader_device_scope']
def test_mapping_and_restore(monkeypatch):
 monkeypatch.setenv('GPUNETIO_DEVICE_API_RDMA','old')
 with scope('bond0,bond1',1) as actual:
  assert actual=='bond1' and os.environ['GPUNETIO_DEVICE_API_RDMA']=='bond1'
 assert os.environ['GPUNETIO_DEVICE_API_RDMA']=='old'
def test_missing_and_invalid(monkeypatch):
 import pytest
 monkeypatch.delenv('GPUNETIO_DEVICE_API_RDMA',raising=False)
 with scope(None,0) as actual: assert actual is None
 with pytest.raises(ValueError):
  with scope('bond0,',1):pass
 assert 'GPUNETIO_DEVICE_API_RDMA' not in os.environ
