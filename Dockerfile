# 平台合规要求：官方轻量基础镜像 + EXPOSE 端口 + /health 端点（见 spec §4.9）
# BASE_IMAGE 可通过 --build-arg 覆盖，便于本地无法访问平台 registry 时用公共镜像验证构建
ARG BASE_IMAGE=modelhubxc-4pd.tencentcloudcr.com/xc_agent_platform/python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# 平台界面不提供设置环境变量的入口，能注入的只有它自己的 EXTERNAL_SERVICE_TOKEN
# 与 STRATEGY_ID。因此所有可调参数的默认值只能在这里给——**绝不要把凭据写进来**，
# 镜像层是明文且会随仓库分发。
#
# 存储：平台不挂载卷，/data 不存在。放在镜像自带目录，容器重启后数据会丢；
# 启动时会从平台认领仍在途的任务（见 main._adopt_in_flight_tasks），
# 认领不成功则拉闸暂停提交，避免丢库导致重复提交。
ENV STORAGE_PATH=/app/data/agent.db

# 演练已完成（本机对真实平台跑通：去重、选卡、vllm 与 llamacpp 两种 configParams
# 均已逐字核对）。现在真实提交。
# 需要回到演练模式排查问题时，把这里改回 true 再发一个 tag。
ENV DRY_RUN=false

# ── 平台凭据：绝不写在这里 ────────────────────────────────────────────────
# 这里曾经硬编码过一个 xcToken，随公开仓库泄露过一次（令牌已作废）。
#
# 镜像层是明文的，`docker history` / `docker inspect` 都能读出来；写进 Dockerfile
# 还会同时进入 git 历史，**改掉之后依然留在历史里**。仓库又是公开的，等于直接公布。
#
# 令牌只能在运行时进入进程：平台注入，或本地 `docker run -e XC_TOKEN=...`
# （见 scripts/dry-run-local.sh，它用隐藏输入读取，不进 shell history）。
# tests/test_no_secrets_in_image.py 会拦住再次写入的尝试。

# 平台内网连不上 huggingface.co（线上实测 connect timeout），默认关闭该来源，
# 只用国内可达的 ModelScope。若你的环境有 HF 镜像，设 HF_ENDPOINT 后再打开。
ENV HF_DISCOVERY_ENABLED=false
ENV MODELSCOPE_DISCOVERY_ENABLED=true

RUN mkdir -p /app/data

EXPOSE 8080

# 通过 python 直启以保证 SIGTERM 传递到进程（不要用 shell 形式）
ENTRYPOINT ["auto-adapter"]
