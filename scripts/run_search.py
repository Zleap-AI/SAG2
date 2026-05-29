import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.modules.search.searcher import SAGSearcher
from pipeline.modules.search.config import (
    SearchConfig, RerankStrategy, RerankConfig
)
from pipeline.utils import get_logger
from pipeline.core.prompt.manager import get_prompt_manager

logger = get_logger("scripts.run_search_only")


def get_dataset_path_for_name(dataset_name: str) -> str:
    """根据数据集名称获取默认数据集路径"""
    return f"./pipeline/evaluation/dataset/{dataset_name}.json"


def get_latest_source_config_id(dataset_name: str) -> str:
    """
    根据数据集名称从最新的 source_info.json 获取 source_config_id
    """
    import re
    import os
    from dotenv import load_dotenv

    load_dotenv()

    source_base = Path("./pipeline/evaluation/source/SAG")
    timestamp_pattern = re.compile(r'^\d{8}_\d{6}$')
    model_name = os.getenv('LLM_MODEL') or os.getenv('MULTIMODAL_LLM_MODEL')

    if model_name:
        model_dir = source_base / model_name / dataset_name
        if model_dir.exists():
            timestamp_dirs = [d for d in model_dir.iterdir() if d.is_dir() and timestamp_pattern.match(d.name)]
            if timestamp_dirs:
                latest_dir = max(timestamp_dirs, key=lambda d: d.name)
                source_info_path = latest_dir / "source_info.json"
                if source_info_path.exists():
                    with open(source_info_path, 'r', encoding='utf-8') as f:
                        source_info = json.load(f)
                    return source_info.get('source_config_id')

    if source_base.exists():
        all_source_infos = []
        for model_dir in source_base.iterdir():
            if not model_dir.is_dir():
                continue
            dataset_dir = model_dir / dataset_name
            if not dataset_dir.exists():
                continue
            timestamp_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and timestamp_pattern.match(d.name)]
            for ts_dir in timestamp_dirs:
                source_info_path = ts_dir / "source_info.json"
                if source_info_path.exists():
                    all_source_infos.append((ts_dir.name, source_info_path))

        if all_source_infos:
            latest_timestamp, latest_path = max(all_source_infos, key=lambda x: x[0])
            with open(latest_path, 'r', encoding='utf-8') as f:
                source_info = json.load(f)
            return source_info.get('source_config_id')

    return None


async def search_single_query(
    searcher: SAGSearcher,
    strategy: str,
    query: str,
    source_config_ids: List[str],
    top_k: int = 10,
) -> Dict:
    """执行单个查询检索"""
    strategy_map = {
        "atomic": RerankStrategy.ATOMIC,
        "multi": RerankStrategy.MULTI,
        "vector": RerankStrategy.VECTOR,
    }

    rerank_config = RerankConfig(
        strategy=strategy_map.get(strategy, RerankStrategy.MULTI),
        top_k=top_k,
    )

    search_config = SearchConfig(
        query=query,
        source_config_ids=source_config_ids,
        rerank=rerank_config,
    )

    result = await searcher.search(search_config)
    return result


def format_section_text(section: Dict) -> str:
    """格式化段落为 "标题\n内容" 格式"""
    heading = section.get("heading", "")
    content = section.get("content", "")
    # 确保 heading 以 "# " 开头（如果不为空）
    if heading and not heading.startswith("#"):
        heading = f"# {heading}"
    if heading and content:
        return f"{heading}\n{content}"
    elif heading:
        return heading
    elif content:
        return content
    return ""


async def main():
    parser = argparse.ArgumentParser(description="单独的搜索脚本 - 只执行检索")
    
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["atomic", "multi", "vector"],
        help="搜索策略"
    )
    
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="数据集名称"
    )
    
    parser.add_argument(
        "--source-config-ids",
        type=str,
        help="信息源ID列表（逗号分隔），不指定则自动从数据集名称查找"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="输出根目录（默认: output）"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="返回前K个结果（默认: 10）"
    )
    
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="搜索并发数（默认: 1）"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理的数据集条数"
    )
    
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="起始索引（从0开始）"
    )
    
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="结束索引"
    )
    
    args = parser.parse_args()
    
    # 确定 source_config_ids
    if args.source_config_ids:
        source_config_ids = [s.strip() for s in args.source_config_ids.split(",")]
        print(f"使用指定的信息源: {source_config_ids}")
    else:
        source_config_id = get_latest_source_config_id(args.dataset_name)
        if not source_config_id:
            print(f"✗ 未找到数据集 '{args.dataset_name}' 的数据源")
            print(f"  请先运行: python scripts/run_pipeline.py upload --dataset-name {args.dataset_name}")
            return 1
        source_config_ids = [source_config_id]
        print(f"自动找到信息源: {source_config_id}")
    
    # 确定数据集路径
    dataset_path = get_dataset_path_for_name(args.dataset_name)
    if not Path(dataset_path).exists():
        print(f"✗ 数据集文件不存在: {dataset_path}")
        return 1
    
    # 加载数据集
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # 应用范围限制
    start_idx = args.start
    end_idx = args.end if args.end is not None else len(dataset)
    if args.limit:
        end_idx = min(start_idx + args.limit, end_idx)
    
    dataset = dataset[start_idx:end_idx]
    
    print(f"\n{'=' * 70}")
    print(f"搜索配置:")
    print(f"  策略: {args.strategy}")
    print(f"  数据集: {args.dataset_name}")
    print(f"  信息源: {source_config_ids}")
    print(f"  问题范围: [{start_idx}:{end_idx}] (共 {len(dataset)} 个)")
    print(f"  Top-K: {args.top_k}")
    print(f"  并发数: {args.max_concurrency}")
    print(f"{'=' * 70}\n")
    
    # 创建输出目录（按时间戳和数据集名称）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = Path(args.output_dir) / "search_results" / args.dataset_name / timestamp
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # 初始化搜索器
    prompt_manager = get_prompt_manager()
    searcher = SAGSearcher(prompt_manager=prompt_manager)
    
    # 执行搜索
    semaphore = asyncio.Semaphore(args.max_concurrency)
    results = []
    
    async def search_one(idx: int, item: Dict):
        async with semaphore:
            i = start_idx + idx + 1
            question = item.get("question", "")
            print(f"[{i}/{start_idx + len(dataset)}] 搜索: {question[:60]}...")
            
            query_start = time.perf_counter()
            search_result = await search_single_query(
                searcher=searcher,
                strategy=args.strategy,
                query=question,
                source_config_ids=source_config_ids,
                top_k=args.top_k,
            )
            query_time = time.perf_counter() - query_start
            
            sections = search_result.get("sections", [])
            retrieved_docs = [format_section_text(sec) for sec in sections[:args.top_k]]
            
            result = {
                "question_index": i,
                "question": question,
                "retrieved_docs": retrieved_docs,
            }
            
            print(f"  ✓ 检索到 {len(retrieved_docs)} 个结果，耗时: {query_time:.2f}s")
            return result
    
    # 并发执行
    tasks = [search_one(idx, item) for idx, item in enumerate(dataset)]
    results = await asyncio.gather(*tasks)
    
    # 保存结果
    output_file = output_subdir / f"search_results_{args.dataset_name}_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 同时保存一份为 search_results.json（方便查找最新）
    latest_file = output_subdir / "search_results.json"
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"✅ 搜索完成!")
    print(f"  输出目录: {output_subdir}")
    print(f"  结果文件: {output_file}")
    print(f"  最新结果: {latest_file}")
    print(f"{'=' * 70}\n")
    
    # 打印示例结果
    if results:
        print("示例结果（第一个问题）:")
        print(json.dumps(results[0], ensure_ascii=False, indent=2)[:1000] + "...")
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)