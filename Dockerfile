# 平台合规要求：官方轻量基础镜像 + EXPOSE 端口 + /health 端点（见 spec §4.9）
FROM modelhubxc-4pd.tencentcloudcr.com/xc_agent_platform/python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

EXPOSE 8080

# 通过 python 直启以保证 SIGTERM 传递到进程（不要用 shell 形式）
ENTRYPOINT ["auto-adapter"]
