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

# 首次上线先演练：完整跑发现/去重/配置生成，但不真的提交。
# 验证通过后把这一行改成 false（或删掉）再发一个新 tag。
ENV DRY_RUN=true

# 平台内网连不上 huggingface.co（线上实测 connect timeout），默认关闭该来源，
# 只用国内可达的 ModelScope。若你的环境有 HF 镜像，设 HF_ENDPOINT 后再打开。
ENV HF_DISCOVERY_ENABLED=false
ENV MODELSCOPE_DISCOVERY_ENABLED=true

RUN mkdir -p /app/data

EXPOSE 8080

# 通过 python 直启以保证 SIGTERM 传递到进程（不要用 shell 形式）
ENTRYPOINT ["auto-adapter"]
