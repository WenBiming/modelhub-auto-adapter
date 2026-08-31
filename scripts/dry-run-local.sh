#!/usr/bin/env bash
# 本机演练：用你自己的 xcToken 跑一轮完整流水线，但不向平台提交任何任务。
#
# 令牌通过隐藏输入读取，不进 shell history、不进文件、不进镜像。
# 平台上跑不通的鉴权不影响这一步——这里用的是你个人的开放平台 xcToken。
set -euo pipefail

IMAGE=${IMAGE:-auto-adapter}
MINUTES=${MINUTES:-3}
NAME=aa-dryrun-$$

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "构建镜像 $IMAGE ..."
  docker build --build-arg BASE_IMAGE=python:3.11-slim -t "$IMAGE" .
fi

read -rsp "粘贴你的 xcToken（输入不会显示）: " XC_TOKEN
echo
[ -n "$XC_TOKEN" ] || { echo "没有输入令牌，退出"; exit 1; }

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# DRY_RUN 由镜像的 ENV 默认开启，这里再显式写一次，免得镜像被改过而不自知。
docker run -d --name "$NAME" \
  -e XC_TOKEN="$XC_TOKEN" \
  -e STRATEGY_ID="local-dry-run" \
  -e DRY_RUN=true \
  -e TICK_SECONDS=60 \
  "$IMAGE" >/dev/null
unset XC_TOKEN

echo "已启动，观察 ${MINUTES} 分钟（Ctrl-C 可提前结束）..."
echo "----------------------------------------------------------------"

# 不用 timeout：macOS 自带的 BSD 工具集里没有它（那是 GNU coreutils）。
# 改为后台跟随日志 + 到点 docker stop：容器收到真实 SIGTERM 优雅退出后，
# docker logs -f 自然结束——顺便把停机路径也验了。
( docker logs -f "$NAME" 2>&1 \
    | grep -Ev "werkzeug|Serving Flask|Debug mode|Running on|WARNING: This|Press CTRL" ) &

sleep "$((MINUTES * 60))"
echo "----------------------------------------------------------------"
echo "发送 SIGTERM，等待优雅停机..."
START=$(date +%s)
docker stop -t 35 "$NAME" >/dev/null 2>&1 || true
echo "退出码 $(docker inspect "$NAME" --format '{{.State.ExitCode}}' 2>/dev/null || echo '?')"\
     "，耗时 $(( $(date +%s) - START ))s（平台给 30s 宽限）"
wait 2>/dev/null || true
echo "结束。容器已清理。"
