#!/bin/bash
# SAG2 数据库初始化脚本
# 在现有MySQL服务中创建sag2数据库、用户并初始化表结构

set -e

echo "=========================================="
echo "  SAG2 数据库初始化"
echo "=========================================="
echo ""

# 从 .env 文件加载配置（脚本所在目录的上级，即项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"
if [ -f "${ENV_FILE}" ]; then
    echo "加载配置文件: ${ENV_FILE}"
    # 只导出非注释、非空行的变量
    set -a
    # shellcheck disable=SC1090
    source <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "${ENV_FILE}")
    set +a
else
    echo "警告: 未找到 .env 文件 (${ENV_FILE})，将使用默认值"
fi

# 数据库连接信息（使用root用户创建数据库和用户）
# Docker容器名称和root凭据不在 .env 中，保留环境变量或默认值
MYSQL_CONTAINER=${MYSQL_CONTAINER:-sag2_mysql}
MYSQL_ROOT_USER=${MYSQL_ROOT_USER:-root}
MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD:-sag2_root}
# 以下变量优先读取 .env 中的值
MYSQL_HOST=${MYSQL_HOST:-localhost}
MYSQL_PORT=${MYSQL_PORT:-3306}
MYSQL_DATABASE=${MYSQL_DATABASE:-sag2}
MYSQL_USER=${MYSQL_USER:-sag2}
MYSQL_PASSWORD=${MYSQL_PASSWORD:-sag2}

# 检测MySQL连接方式
if docker ps | grep -q ${MYSQL_CONTAINER}; then
    echo "检测到MySQL Docker容器: ${MYSQL_CONTAINER}"
    MYSQL_CMD="docker exec -i ${MYSQL_CONTAINER} mysql -h 127.0.0.1 -u ${MYSQL_ROOT_USER} -p${MYSQL_ROOT_PASSWORD}"
    MYSQL_USER_CMD="docker exec -i ${MYSQL_CONTAINER} mysql -h 127.0.0.1 -u ${MYSQL_USER} -p${MYSQL_PASSWORD}"
elif command -v mysql &> /dev/null; then
    echo "使用本地MySQL客户端"
    MYSQL_CMD="mysql -h ${MYSQL_HOST} -P ${MYSQL_PORT} -u ${MYSQL_ROOT_USER} -p${MYSQL_ROOT_PASSWORD}"
    MYSQL_USER_CMD="mysql -h ${MYSQL_HOST} -P ${MYSQL_PORT} -u ${MYSQL_USER} -p${MYSQL_PASSWORD}"
else
    echo "错误: 未找到MySQL客户端或Docker容器"
    echo "请确保："
    echo "  1. MySQL Docker容器正在运行，或"
    echo "  2. 安装了MySQL客户端工具"
    exit 1
fi

echo "步骤 1/3: 创建数据库和用户"
${MYSQL_CMD} <<EOF
CREATE DATABASE IF NOT EXISTS ${MYSQL_DATABASE} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS '${MYSQL_USER}'@'localhost' IDENTIFIED BY '${MYSQL_PASSWORD}';
CREATE USER IF NOT EXISTS '${MYSQL_USER}'@'%' IDENTIFIED BY '${MYSQL_PASSWORD}';
GRANT ALL PRIVILEGES ON ${MYSQL_DATABASE}.* TO '${MYSQL_USER}'@'localhost';
GRANT ALL PRIVILEGES ON ${MYSQL_DATABASE}.* TO '${MYSQL_USER}'@'%';
FLUSH PRIVILEGES;
EOF
echo "✓ 数据库和用户创建完成"
echo ""
