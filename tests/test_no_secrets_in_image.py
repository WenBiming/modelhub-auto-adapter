"""镜像里不得含凭据。

这条曾经被违反过：Dockerfile 里硬编码的 xcToken 随公开仓库泄露。镜像层是明文，
`docker history` 就能读出来，而且写进 Dockerfile 会同时进入 git 历史——改掉之后
依然留在历史里。这个测试是防止再犯的闸门。
"""
import re
from pathlib import Path

import pytest

DOCKERFILE = Path(__file__).resolve().parent.parent / "Dockerfile"

# 名字看着像凭据的环境变量
_SECRET_NAME = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|APIKEY|API_KEY|CREDENTIAL)",
                          re.IGNORECASE)
# 值看着像凭据：长度够且不是显然的占位符/布尔/路径
_PLACEHOLDER = re.compile(r"^(|true|false|0|1|yes|no|/.*|\$\{.*\}|<.*>|\.\.\.)$",
                          re.IGNORECASE)


def _env_assignments(text: str):
    for line in text.splitlines():
        line = line.strip()
        if not line.upper().startswith("ENV "):
            continue
        body = line[4:].strip()
        if "=" not in body:
            continue
        name, _, value = body.partition("=")
        yield name.strip(), value.strip().strip('"').strip("'")


def test_dockerfile_has_no_hardcoded_credentials():
    for name, value in _env_assignments(DOCKERFILE.read_text()):
        if not _SECRET_NAME.search(name):
            continue
        assert _PLACEHOLDER.match(value) or len(value) < 12, (
            f"Dockerfile 里的 ENV {name} 看起来是一个真实凭据。镜像层是明文，"
            f"而且会进入 git 历史（改掉也留在历史里）。令牌只能在运行时注入："
            f"平台注入，或 docker run -e {name}=...（见 scripts/dry-run-local.sh）。"
        )


@pytest.mark.parametrize("line,should_fail", [
    ("ENV XC_TOKEN=751bbddc7b7b48428375909ab9d8824b", True),   # 真实泄露过的形态
    ("ENV EXTERNAL_SERVICE_TOKEN=abcdef0123456789", True),
    ("ENV DRY_RUN=false", False),
    ("ENV STORAGE_PATH=/app/data/agent.db", False),
    ("ENV AUTH_HEADER=Xc-Token", False),                        # 头名不是凭据
])
def test_detector_recognises_the_shapes_it_must_catch(line, should_fail):
    caught = any(
        _SECRET_NAME.search(n) and not (_PLACEHOLDER.match(v) or len(v) < 12)
        for n, v in _env_assignments(line))
    assert caught is should_fail
