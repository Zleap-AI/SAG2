"""
数据库初始化脚本

一次性完成所有数据库初始化工作：
  1. 创建/确认 10 个必需的表
  2. 插入默认实体类型（Upsert）
  3. 验证表结构与数据完整性

10 个必需的表及关系：
  source_config                                   # 信息源配置（根表）
  ├── article            → source_config          # 文章
  │   ├── article_section → article               # 文章片段
  │   └── entity_type    → article（scope=article）# 文章级实体类型
  ├── entity_type        → source_config（可选）  # 实体类型定义（scope=source/global）
  │   └── entity         → entity_type, source_config  # 实体
  │       └── event_entity → entity, source_event # 事项-实体关联（多对多）
  │           └── event_entity_embedding → event_entity  # 向量（一对一）
  ├── source_event       → source_config, article（可选），self（层级）  # 事项
  ├── source_chunk       → source_config, article（可选）  # 来源片段
  └── kb_document                                 # 知识库文档（独立，无外键）

安全性：
- 使用 Base.metadata.create_all(checkfirst=True) 幂等建表，不删除旧数据
- 实体类型 Upsert：按 type 字段检查，存在则更新，不存在则插入

source_config_id 通过 evaluation/source/{model}/{dataset}/{timestamp}/source_info.json
直接读取，无需 dataset_name / dataset_version 辅助字段。
"""

import argparse
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import select, text

from pipeline.core.config.settings import get_settings
from pipeline.db import EntityType, get_session_factory
from pipeline.db.base import Base, get_engine
from pipeline.models.entity import DEFAULT_ENTITY_TYPES

# 10 个必需的表（与 drop_unused_tables.py 保持一致）
REQUIRED_TABLES = {
    "source_config",
    "article",
    "article_section",
    "kb_document",
    "source_chunk",
    "source_event",
    "event_entity",
    "event_entity_embedding",
    "entity",
    "entity_type",
}


# ─────────────────────────── 输出辅助 ───────────────────────────

def print_header(text_: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text_}")
    print("=" * 70)


def print_success(text_: str) -> None:
    print(f"  ✓ {text_}")


def print_info(text_: str) -> None:
    print(f"  • {text_}")


def print_warning(text_: str) -> None:
    print(f"  ⚠️  {text_}")


def print_error(text_: str) -> None:
    print(f"  ✗ {text_}")


# ─────────────────────────── 命令行参数 ───────────────────────────

def _read_root_password_from_compose() -> str:
    """
    从项目根目录的 docker-compose.yml 中读取 MYSQL_ROOT_PASSWORD。
    仅做简单的 key: value 文本匹配，不依赖 PyYAML，避免额外依赖。
    返回找到的值，找不到则返回空字符串。
    """
    compose_file = project_root / "docker-compose.yml"
    if not compose_file.exists():
        return ""
    for line in compose_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("MYSQL_ROOT_PASSWORD:"):
            value = stripped.split(":", 1)[1].strip()
            # 去除可能的行内注释（如 "sag2_root  # comment"）
            value = value.split("#")[0].strip()
            return value
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pipeline 数据库初始化")
    parser.add_argument(
        "--fix-grants", action="store_true",
        help="使用 MySQL root 账号自动修复用户授权（解决 Access Denied / 172.x 访问被拒问题）",
    )
    parser.add_argument(
        "--mysql-root-password",
        default=None,   # None 表示"未显式传入"，后续自动从 docker-compose.yml 读取
        metavar="PASSWORD",
        help=(
            "MySQL root 密码（用于 --fix-grants）。"
            "未指定时自动读取 docker-compose.yml 中的 MYSQL_ROOT_PASSWORD"
        ),
    )
    return parser.parse_args()


# ─────────────────────────── Step 0（可选）：修复 MySQL 授权 ───────────────────────────

async def fix_mysql_grants(root_password: str) -> None:
    """
    用 root 账号为应用用户授予 MySQL 权限。

    解决场景：Docker 网络中 MySQL 看到的客户端 IP 是容器网关（如 172.31.4.1），
    当 MySQL 以 --skip-name-resolve 启动时，'%' 通配符对 IP 地址不生效，
    必须显式对实际网段（如 '172.%.%.%'）单独授权。

    实现要点：
    - 用 aiomysql 原生连接（autocommit=True），每条 DDL 立即生效
    - 查询 information_schema.processlist 获取 root 连接的实际来源 IP，
      自动推断需要授权的网段（取前两段，如 172.31 → '172.31.%.%'）
    - 清理历史遗留的错误 host='%%' 记录（aiomysql cursor 把 % 转义成 %% 导致）
    """
    print_header("Step 0 / 3  修复 MySQL 用户授权")

    import aiomysql

    settings = get_settings()
    db   = settings.mysql_database
    user = settings.mysql_user
    pwd  = settings.mysql_password

    try:
        conn = await aiomysql.connect(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user="root",
            password=root_password,
            autocommit=True,   # 每条 DDL 立即提交，无事务时序问题
        )
        try:
            async with conn.cursor() as cur:
                # ── 0. 清理历史遗留的错误 host='%%' 记录 ──────────────────
                # 之前脚本用 aiomysql cursor 的 f-string 写了 '%%'，
                # aiomysql 不做转义，结果真的创建了 host='%%' 的用户，需删除。
                await cur.execute(
                    f"SELECT COUNT(*) FROM mysql.user "
                    f"WHERE user='{user}' AND host='%%'"
                )
                row = await cur.fetchone()
                if row and row[0] > 0:
                    await cur.execute(f"DROP USER IF EXISTS '{user}'@'%%'")
                    print_info(f"已清理错误记录：{user}@'%%'")

                # ── 1. 确保数据库存在 ──────────────────────────────────────
                await cur.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{db}` "
                    f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )

                # ── 2. 检测 skip_name_resolve 模式 ────────────────────────
                await cur.execute("SELECT @@skip_name_resolve")
                row = await cur.fetchone()
                skip_name_resolve = bool(row and row[0])

                # ── 3. 确定需要授权的 host 列表 ───────────────────────────
                # 基础：'localhost'（UNIX socket）+ '%'（通用通配符）
                hosts = ["localhost", "%"]

                if skip_name_resolve:
                    # --skip-name-resolve 模式下 '%' 对 IP 地址不生效，
                    # 需要查出当前连接的实际客户端 IP，对其所在 /16 网段授权。
                    await cur.execute(
                        "SELECT host FROM information_schema.processlist "
                        "WHERE user = 'root' LIMIT 1"
                    )
                    row = await cur.fetchone()
                    if row:
                        # host 格式为 "172.31.4.1:46010"，取 IP 部分
                        client_ip = row[0].split(":")[0]
                        parts = client_ip.split(".")
                        if len(parts) == 4:
                            # 取前两段，覆盖同一 Docker bridge 网络的所有容器
                            ip_pattern = f"{parts[0]}.{parts[1]}.%.%"
                            if ip_pattern not in hosts:
                                hosts.append(ip_pattern)
                                print_info(
                                    f"检测到 --skip-name-resolve 模式，"
                                    f"将额外授权网段：'{ip_pattern}'"
                                )

                # ── 4. 为每个 host 创建用户并授权 ────────────────────────
                for host in hosts:
                    await cur.execute(
                        f"CREATE USER IF NOT EXISTS '{user}'@'{host}' "
                        f"IDENTIFIED BY '{pwd}'"
                    )
                    await cur.execute(
                        f"GRANT ALL PRIVILEGES ON `{db}`.* TO '{user}'@'{host}'"
                    )

                await cur.execute("FLUSH PRIVILEGES")

        finally:
            conn.close()

        print_success(
            f"已授权：{user}@{hosts} → 数据库 {db}.*"
        )
    except Exception as grant_err:
        err_str = str(grant_err)
        if "1045" in err_str or "Access denied" in err_str:
            raise RuntimeError(
                f"root 账号连接失败（密码错误？）：{grant_err}\n"
                f"  请确认 --mysql-root-password 的值正确"
            ) from grant_err
        raise


# ─────────────────────────── Step 1：建表 ───────────────────────────

def get_project_tables() -> set[str]:
    """返回 ORM 中已注册的表名集合"""
    from pipeline.db import models  # noqa: F401  触发 ORM 注册
    tables = set()
    for table_name in Base.metadata.tables.keys():
        tables.add(table_name.split(".")[-1] if "." in table_name else table_name)
    return tables


async def create_tables() -> None:
    """
    幂等建表：仅创建不存在的表，不删除任何数据。
    """
    print_header("Step 1 / 3  创建表结构")

    from pipeline.db import models  # noqa: F401

    orm_tables = get_project_tables()
    print_info(f"ORM 定义了 {len(orm_tables)} 个表：{sorted(orm_tables)}")

    # 验证 ORM 定义与期望一致
    missing_in_orm = REQUIRED_TABLES - orm_tables
    extra_in_orm   = orm_tables - REQUIRED_TABLES
    if missing_in_orm:
        print_warning(f"ORM 缺少以下必需表定义：{missing_in_orm}")
    if extra_in_orm:
        print_warning(f"ORM 含有额外表（非必需）：{extra_in_orm}")

    engine = get_engine()
    async with engine.begin() as conn:
        # checkfirst=True：表已存在则跳过，不会丢数据
        await conn.run_sync(Base.metadata.create_all, checkfirst=True)

    print_success(f"建表完成（已存在的表自动跳过）")

    # 检查数据库实际表列表
    async with engine.connect() as conn:
        result = await conn.execute(text("SHOW TABLES"))
        actual_tables = {row[0] for row in result}

    missing_in_db = REQUIRED_TABLES - actual_tables
    if missing_in_db:
        print_error(f"以下必需表仍未在数据库中：{missing_in_db}")
        raise RuntimeError(f"建表失败，缺少：{missing_in_db}")

    for t in sorted(REQUIRED_TABLES):
        print_success(f"{t}")


# ─────────────────────────── Step 2：插入默认实体类型 ───────────────────────────

async def insert_default_entity_types() -> None:
    """
    Upsert 默认实体类型。
    - 按 type 字段检查是否存在（is_default=True）
    - 存在：比对字段，有变化则更新（保留原 ID，不破坏外键）
    - 不存在：插入新记录
    """
    print_header("Step 2 / 3  同步默认实体类型")

    factory = get_session_factory()
    async with factory() as session:
        inserted = updated = unchanged = 0

        for type_def in DEFAULT_ENTITY_TYPES:
            result = await session.execute(
                select(EntityType).where(
                    EntityType.type == type_def.type,
                    EntityType.is_default == True,  # noqa: E712
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                needs_update = (
                    existing.name                != type_def.name
                    or existing.description      != type_def.description
                    or existing.weight           != type_def.weight
                    or existing.similarity_threshold != type_def.similarity_threshold
                )
                if needs_update:
                    existing.name                 = type_def.name
                    existing.description          = type_def.description
                    existing.weight               = type_def.weight
                    existing.similarity_threshold = type_def.similarity_threshold
                    existing.is_active            = type_def.is_active
                    print_success(f"{type_def.type} ({type_def.name}): 已更新")
                    updated += 1
                else:
                    print_info(f"{type_def.type} ({type_def.name}): 无变化，跳过")
                    unchanged += 1
                continue

            session.add(EntityType(
                id                   = type_def.id,
                source_config_id     = type_def.source_config_id,
                type                 = type_def.type,
                name                 = type_def.name,
                is_default           = type_def.is_default,
                description          = type_def.description,
                weight               = type_def.weight,
                similarity_threshold = type_def.similarity_threshold,
                extra_data           = None,
                is_active            = type_def.is_active,
            ))
            print_success(f"{type_def.type} ({type_def.name}): 插入成功")
            inserted += 1

        await session.commit()

    print_header("同步总结")
    if inserted:   print_success(f"新插入: {inserted} 个")
    if updated:    print_success(f"已更新: {updated} 个")
    if unchanged:  print_info(f"无变化: {unchanged} 个")


# ─────────────────────────── Step 3：验证 ───────────────────────────

async def verify_database() -> None:
    """验证数据库最终状态：表存在性 + 实体类型数量。"""
    print_header("Step 3 / 3  验证数据库")

    engine = get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(text("SHOW TABLES"))
        actual_tables = {row[0] for row in result}

    missing = REQUIRED_TABLES - actual_tables
    extra   = actual_tables   - REQUIRED_TABLES

    print_info(f"数据库现有 {len(actual_tables)} 个表")
    if missing:
        print_error(f"缺少必需表：{missing}")
    if extra:
        print_warning(f"存在额外表（非本项目必需）：{extra}")
    if not missing:
        print_success(f"10 个必需表全部就绪")

    # 实体类型
    factory = get_session_factory()
    async with factory() as session:
        result     = await session.execute(select(EntityType))
        entity_types = result.scalars().all()

    if entity_types:
        print_info(f"默认实体类型共 {len(entity_types)} 个：")
        for et in entity_types:
            print_success(
                f"{et.type} ({et.name}): "
                f"weight={et.weight}, threshold={et.similarity_threshold}"
            )
    else:
        print_warning("未找到任何实体类型")


# ─────────────────────────── 主流程 ───────────────────────────

async def main() -> None:
    args = parse_args()
    try:
        print_header("pipeline 数据库初始化")
        print_info("将依次执行：建表 → 插入默认数据 → 验证")

        # Step 0（可选）：修复 MySQL 用户授权
        if args.fix_grants:
            # 优先用命令行传入的密码，其次自动从 docker-compose.yml 读取
            root_pwd = args.mysql_root_password or _read_root_password_from_compose()
            if not root_pwd:
                print_error(
                    "--fix-grants 需要 root 密码，请确保项目根目录存在 docker-compose.yml "
                    "且其中包含 MYSQL_ROOT_PASSWORD，或通过 --mysql-root-password=<密码> 手动传入"
                )
                sys.exit(1)
            await fix_mysql_grants(root_pwd)
            # 授权完成后重置应用 engine 单例，确保 Step 1 的连接池在权限
            # 完全生效后重新建立，避免使用授权前缓存的旧连接
            from pipeline.db.base import reset_engine
            reset_engine()

        await create_tables()
        await insert_default_entity_types()
        await verify_database()

        print_header("初始化完成")
        print_success("所有步骤执行成功！")
        print("=" * 70 + "\n")

    except Exception as e:
        print_error(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        from pipeline.db.base import close_database
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
