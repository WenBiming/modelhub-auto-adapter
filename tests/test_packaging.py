"""打包回归：configParams 模板必须是 auto_adapter 包的一部分。

容器用 `pip install .`（非 editable）安装，模板若没被 package-data 收进 wheel，
config_gen.render_config_params 会在生产环境第一次渲染时 FileNotFoundError，
整个 tick 在 submitter/monitor/failure 之前就中断——而开发环境的 editable 安装
永远发现不了。importlib.resources 的解析路径与"已安装的包"一致，因此能捕获
package-data 缺失。
"""
from importlib.resources import files

import pytest
import yaml

from auto_adapter import config_gen


@pytest.mark.parametrize("template", ["vllm.yaml", "transformers.yaml"])
def test_template_is_readable_as_package_resource(template):
    resource = files("auto_adapter").joinpath(f"templates/{template}")
    assert resource.is_file(), f"{template} is not packaged with auto_adapter"
    assert resource.read_text().strip(), f"{template} is packaged but empty"


@pytest.mark.parametrize("framework", ["vllm", "transformers"])
def test_render_config_params_works_from_packaged_templates(framework):
    """端到端：渲染必须真的产出可解析的 YAML，而不只是文件存在。"""
    cfg = yaml.safe_load(config_gen.render_config_params(framework, tp_size=1))
    assert cfg["framework"] == framework
