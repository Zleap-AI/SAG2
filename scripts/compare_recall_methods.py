#!/usr/bin/env python3
"""
使用方法 B（原始字符串精确匹配）对比两个检索结果文件的 Recall@K

方法 B（cross_validation.py）：
  - 将 GT 构建为原始字符串 "title\ncontent"
  - 直接用 Python == 做精确字符串匹配，无任何标准化
  - 用 set 去重，同一 GT 段落不重复计分

用法：
  # 对比两个检索结果
  uv run python scripts/compare_recall_methods.py \
      --predictions output/results_a.json output/results_b.json \
      --dataset-name musique \
      --k-values 1,2,3,5,10

  # 只看前 50 条，快速验证
  uv run python scripts/compare_recall_methods.py \
      --predictions output/results_a.json output/results_b.json \
      --dataset-name musique \
      --limit 50

  # 评估单个检索结果
  uv run python scripts/compare_recall_methods.py \
      --predictions output/results.json \
      --dataset-name musique


      uv run python scripts/compare_recall_methods.py \
    --predictions output/musique/multi1/20260523_133043/search_results.json output/musique/multi/20260523_133317/search_results.json \
    --dataset-name musique --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.evaluation.utils import normalize_text, extract_sentences


def load_predictions(predictions_path: str) -> List[dict]:
    """加载检索结果，统一为 {query_id, question, retrieved_docs} 格式"""
    with open(predictions_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    results = []
    for item in raw:
        # 兼容多种 ID 字段名，按优先级取第一个存在的
        qid = item.get("question_index") or item.get("id") or item.get("query_id")
        if qid is None or "retrieved_docs" not in item:
            continue

        results.append({
            "query_id": str(qid),
            "question": item.get("question", ""),
            "retrieved_docs": item["retrieved_docs"],
        })

    print(f"✓ 加载检索结果: {len(results)} 条 ({predictions_path})")
    return results


def load_ground_truth(dataset_path: str) -> List[dict]:
    """
    加载数据集 GT，兼容两种格式：

    Format A — paragraphs + is_supporting（自定义格式）:
      item.paragraphs = [{"title":..., "paragraph_text":..., "is_supporting": true/false}, ...]

    Format B — context + supporting_facts（HotpotQA 原始格式）:
      item.context          = [[title, [sent0, sent1, ...]], ...]
      item.supporting_facts = [[title, sent_idx], ...]

    每条返回：{query_id, question, gt_docs: [{title, content}], gt_strings: ["title\ncontent", ...]}
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ground_truth = []
    for item in data:
        qid = str(item.get("id", "") or item.get("_id", ""))
        if not qid:
            continue

        gt_docs: List[dict] = []
        gt_strings: List[str] = []

        # ── Format A: paragraphs + is_supporting ──────────────────
        if item.get("paragraphs"):
            for para in item["paragraphs"]:
                if para.get("is_supporting", False):
                    title   = para.get("title", "")
                    content = para.get("paragraph_text", "") or para.get("content", "")
                    gt_docs.append({"title": title, "content": content})
                    gt_strings.append(f"{title}\n{content}")

        # ── Format B: context + supporting_facts (HotpotQA) ───────
        elif item.get("context") and item.get("supporting_facts"):
            context_map: Dict[str, List[str]] = {}
            for ctx in item["context"]:
                if not (isinstance(ctx, (list, tuple)) and len(ctx) >= 2):
                    continue
                ctx_sents = ctx[1] if isinstance(ctx[1], list) else [ctx[1]]
                context_map[ctx[0]] = ctx_sents

            seen: Set[str] = set()
            for sf in item["supporting_facts"]:
                if not (isinstance(sf, (list, tuple)) and len(sf) >= 1):
                    continue
                sf_title = sf[0]
                if sf_title in seen or sf_title not in context_map:
                    continue
                seen.add(sf_title)
                sents = context_map[sf_title]
                content = " ".join(s.strip() for s in sents).strip()
                if not (sf_title and content):
                    continue
                gt_docs.append({"title": sf_title, "content": content})
                gt_strings.append(f"{sf_title}\n{content}")

        if gt_docs:
            ground_truth.append({
                "query_id": qid,
                "question": item.get("question", ""),
                "gt_docs": gt_docs,
                "gt_strings": gt_strings,
            })

    print(f"✓ 加载 Ground Truth: {len(ground_truth)} 条 ({dataset_path})")
    return ground_truth


def align_by_question(
    predictions: List[dict],
    ground_truth: List[dict]
) -> List[Tuple[dict, dict]]:
    """
    通过问题文本把 predictions 和 ground truth 对齐。
    先精确匹配，再词汇重叠模糊匹配，均失败则尝试按位置顺序对齐。
    返回配对列表 [(pred, gt), ...]
    """
    # 构建 GT 索引：normalized_question → gt_item
    gt_by_norm: Dict[str, dict] = {}
    for gt in ground_truth:
        norm = normalize_text(gt["question"])
        gt_by_norm[norm] = gt

    paired = []
    unmatched_preds = []

    for pred in predictions:
        norm_pred = normalize_text(pred["question"])

        # 精确匹配
        if norm_pred in gt_by_norm:
            paired.append((pred, gt_by_norm[norm_pred]))
            continue

        # 词汇重叠模糊匹配
        best_gt, best_score = None, 0.0
        pred_words = set(norm_pred.split())
        for norm_gt, gt in gt_by_norm.items():
            if not pred_words or not norm_gt:
                continue
            gt_words = set(norm_gt.split())
            score = len(pred_words & gt_words) / max(len(pred_words), len(gt_words))
            if score > best_score and score > 0.5:
                best_score, best_gt = score, gt

        if best_gt:
            paired.append((pred, best_gt))
        else:
            unmatched_preds.append(pred)

    if unmatched_preds:
        print(f"⚠️  {len(unmatched_preds)} 条查询未能通过问题文本匹配，将尝试按顺序对齐")
        # 构建未匹配的 GT 列表（已配对的 GT query_id）
        matched_gt_ids = {gt["query_id"] for _, gt in paired}
        remaining_gt = [gt for gt in ground_truth if gt["query_id"] not in matched_gt_ids]
        for pred, gt in zip(unmatched_preds, remaining_gt):
            paired.append((pred, gt))

    print(f"✓ 成功配对: {len(paired)} 条")
    return paired


# ──────────────────────────────────────────────────────────────────────────────
# 方法 B：原始字符串精确匹配（cross_validation.py）
# ──────────────────────────────────────────────────────────────────────────────

def recall_method_b(retrieved_docs: List[str], gt_strings: List[str], k: int) -> float:
    """
    方法 B 的 Recall@K：
    直接用 == 比较 retrieved_doc 字符串与 GT "title\ncontent" 字符串，
    无任何标准化；GT 同样按 index 去重。
    """
    top_k = retrieved_docs[:k]
    matched: Set[int] = set()
    for ret in top_k:
        for idx, gt_str in enumerate(gt_strings):
            if idx not in matched and ret == gt_str:
                matched.add(idx)
    return len(matched) / len(gt_strings) if gt_strings else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 单文件评估
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_single(
    paired: List[Tuple[dict, dict]],
    k_values: List[int],
) -> Dict[int, List[float]]:
    """对单个检索结果文件跑方法 B，返回每个 K 的 recall 列表"""
    recalls: Dict[int, List[float]] = {k: [] for k in k_values}

    for pred, gt in paired:
        retrieved = pred["retrieved_docs"]
        gt_strs   = gt["gt_strings"]
        for k in k_values:
            rb = recall_method_b(retrieved, gt_strs, k)
            recalls[k].append(rb)

    return recalls


# ──────────────────────────────────────────────────────────────────────────────
# 逐条详情打印（单文件模式）
# ──────────────────────────────────────────────────────────────────────────────

def print_query_detail_single(
    idx: int,
    total: int,
    pred: dict,
    gt: dict,
    k_values: List[int],
    max_k: int,
) -> None:
    SEP  = "=" * 120
    LINE = "─" * 120

    gt_docs   = gt["gt_docs"]
    gt_strs   = gt["gt_strings"]
    retrieved = pred["retrieved_docs"]
    question  = gt["question"] or pred["question"]

    recalls = {k: recall_method_b(retrieved, gt_strs, k) for k in k_values}

    print(f"\n{SEP}")
    print(f"📝 问题 {idx}/{total}")
    print(LINE)
    print(f"问题: {question}")

    print(f"\n✅ 正确答案 (Ground Truth) - 相关文档 ({len(gt_docs)} 个):")
    for i, doc in enumerate(gt_docs, 1):
        sents   = extract_sentences(doc["content"])
        summary = (doc["content"][:100].rstrip() + "...") if len(doc["content"]) > 100 else doc["content"]
        print(f"   {i}. 标题: {doc['title']}")
        print(f"      内容摘要: {summary}")
        print(f"      句子数: {len(sents)}")

    print(f"\n📊 Recall@K:")
    for k in sorted(k_values):
        print(f"   @{k}:  {recalls[k]:.4f}")

    show_n = min(max_k, len(retrieved))
    print(f"\n🔮 Top {show_n} 检索结果（方法B 精确匹配）:")
    b_matched: Set[int] = set()
    for pos, ret_str in enumerate(retrieved[:max_k], 1):
        lines = ret_str.strip().split('\n')
        title = lines[0] if lines else "(空)"

        b_any_idx, b_new_idx = None, None
        for gi, gt_str in enumerate(gt_strs):
            if ret_str == gt_str:
                if b_any_idx is None:
                    b_any_idx = gi
                if gi not in b_matched and b_new_idx is None:
                    b_new_idx = gi

        if b_new_idx is not None:
            b_matched.add(b_new_idx)

        b_sym = "✅" if b_new_idx is not None else ("🔁" if b_any_idx is not None else "❌")
        print(f"   {pos:2d}.  [{b_sym}]  {title}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# 两文件对比详情打印
# ──────────────────────────────────────────────────────────────────────────────

def print_query_detail_compare(
    idx: int,
    total: int,
    pred_a: dict,
    pred_b: dict,
    gt: dict,
    k_values: List[int],
    max_k: int,
    label_a: str,
    label_b: str,
) -> None:
    SEP  = "=" * 120
    LINE = "─" * 120

    gt_docs   = gt["gt_docs"]
    gt_strs   = gt["gt_strings"]
    ret_a     = pred_a["retrieved_docs"]
    ret_b     = pred_b["retrieved_docs"]
    question  = gt["question"] or pred_a["question"]

    recalls_a = {k: recall_method_b(ret_a, gt_strs, k) for k in k_values}
    recalls_b = {k: recall_method_b(ret_b, gt_strs, k) for k in k_values}
    diff_ks   = {k for k in k_values if abs(recalls_a[k] - recalls_b[k]) > 1e-9}

    print(f"\n{SEP}")
    print(f"📝 问题 {idx}/{total}")
    print(LINE)
    print(f"问题: {question}")

    print(f"\n✅ 正确答案 (Ground Truth) - 相关文档 ({len(gt_docs)} 个):")
    for i, doc in enumerate(gt_docs, 1):
        sents   = extract_sentences(doc["content"])
        summary = (doc["content"][:100].rstrip() + "...") if len(doc["content"]) > 100 else doc["content"]
        print(f"   {i}. 标题: {doc['title']}")
        print(f"      内容摘要: {summary}")
        print(f"      句子数: {len(sents)}")

    print(f"\n📊 Recall@K 对比:")
    w = max(len(label_a), len(label_b), 6)
    for k in sorted(k_values):
        ra = recalls_a[k]
        rb = recalls_b[k]
        marker = " ◄" if k in diff_ks else ""
        print(f"   @{k}:  {label_a:<{w}}={ra:.4f}  {label_b:<{w}}={rb:.4f}{marker}")

    # 检索结果对比（并排展示两个列表）
    show_n = max(min(max_k, len(ret_a)), min(max_k, len(ret_b)))
    print(f"\n🔮 Top {show_n} 检索结果对比（方法B 精确匹配）:")

    matched_a: Set[int] = set()
    matched_b: Set[int] = set()

    def match_info(ret_list, gt_strs, matched_set, pos):
        ret_str = ret_list[pos] if pos < len(ret_list) else None
        if ret_str is None:
            return None, "  ", "(无)"
        lines = ret_str.strip().split('\n')
        title = lines[0] if lines else "(空)"
        b_any_idx, b_new_idx = None, None
        for gi, gt_str in enumerate(gt_strs):
            if ret_str != gt_str:
                continue
            if b_any_idx is None:
                b_any_idx = gi
            if gi not in matched_set and b_new_idx is None:
                b_new_idx = gi
        if b_new_idx is not None:
            matched_set.add(b_new_idx)
        sym = "✅" if b_new_idx is not None else ("🔁" if b_any_idx is not None else "❌")
        return ret_str, sym, title

    for pos in range(show_n):
        _, sym_a, title_a = match_info(ret_a, gt_strs, matched_a, pos)
        _, sym_b, title_b = match_info(ret_b, gt_strs, matched_b, pos)
        print(f"   {pos+1:2d}.  [{label_a}:{sym_a}]  {title_a}")
        print(f"        [{label_b}:{sym_b}]  {title_b}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# 主评估逻辑（单文件）
# ──────────────────────────────────────────────────────────────────────────────

def run_single(
    paired: List[Tuple[dict, dict]],
    k_values: List[int],
    label: str,
    verbose: bool,
) -> None:
    max_k  = max(k_values)
    recalls = evaluate_single(paired, k_values)

    print("\n" + "=" * 80)
    print(f"📊 Recall@K 结果  [{label}]")
    print("=" * 80)
    header = f"{'K':>4}  {'Recall':>10}  {'百分比':>10}"
    print(header)
    print("─" * 40)
    for k in sorted(k_values):
        lst = recalls[k]
        if not lst:
            continue
        avg = sum(lst) / len(lst)
        print(f"  @{k:<3}  {avg:.4f}     ({avg*100:5.1f}%)")

    print(f"\n总查询数: {len(paired)}")

    print(f"\n{'─'*80}")
    print("📈 Recall 分布统计")
    print(f"{'─'*80}")
    for k in sorted(k_values):
        lst = recalls[k]
        if not lst:
            continue
        full    = sum(1 for r in lst if r >= 1.0)
        partial = sum(1 for r in lst if 0 < r < 1.0)
        zero    = sum(1 for r in lst if r == 0.0)
        print(f"  @{k}: full={full:4d}  partial={partial:4d}  zero={zero:4d}")

    if verbose:
        print("\n" + "=" * 120)
        print(f"📝 逐条详情（共 {len(paired)} 条）")
        print("=" * 120)
        for i, (pred, gt) in enumerate(paired, 1):
            print_query_detail_single(i, len(paired), pred, gt, k_values, max_k)

    # ── 0 召回问题列表 ────────────────────────────────────────────────────────
    check_ks = sorted({5, max_k} & set(k_values))   # 只统计 k_values 中存在的 K

    def zero_list(k):
        return [
            (pred, gt) for pred, gt in paired
            if recall_method_b(pred["retrieved_docs"], gt["gt_strings"], k) == 0.0
        ]

    for ck in check_ks:
        zp = zero_list(ck)
        print(f"\n{'─'*80}")
        print(f"🚫 0召回问题统计（@{ck} 下召回率仍为 0）: 共 {len(zp)} 条 / {len(paired)} 条"
              f"（占比 {len(zp)/len(paired)*100:.1f}%）")
        if zp:
            print(f"{'─'*80}")
            for i, (pred, gt) in enumerate(zp, 1):
                question = gt["question"] or pred["question"]
                gt_titles = ", ".join(doc["title"] for doc in gt["gt_docs"])
                print(f"  {i:3d}. 问题: {question}")
                print(f"       GT 标题: {gt_titles}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# 主评估逻辑（两文件对比）
# ──────────────────────────────────────────────────────────────────────────────

def run_compare(
    paired_a: List[Tuple[dict, dict]],
    paired_b: List[Tuple[dict, dict]],
    k_values: List[int],
    label_a: str,
    label_b: str,
    verbose: bool,
) -> None:
    max_k = max(k_values)

    # 按 gt query_id 建立对齐索引
    gt_id_to_pred_b: Dict[str, dict] = {gt["query_id"]: pred for pred, gt in paired_b}

    # 合并对齐：以 paired_a 为基准，找对应的 pred_b
    aligned: List[Tuple[dict, dict, dict]] = []  # (pred_a, pred_b, gt)
    skipped = 0
    for pred_a, gt in paired_a:
        pred_b = gt_id_to_pred_b.get(gt["query_id"])
        if pred_b is None:
            skipped += 1
            continue
        aligned.append((pred_a, pred_b, gt))

    if skipped:
        print(f"⚠️  {skipped} 条在文件B中未找到对应查询，已跳过")
    print(f"✓ 有效对比条数: {len(aligned)}")

    recalls_a: Dict[int, List[float]] = {k: [] for k in k_values}
    recalls_b: Dict[int, List[float]] = {k: [] for k in k_values}
    diff_queries = []

    for pred_a, pred_b, gt in aligned:
        gt_strs = gt["gt_strings"]
        row_diff = False
        ra_row, rb_row = {}, {}
        for k in k_values:
            ra = recall_method_b(pred_a["retrieved_docs"], gt_strs, k)
            rb = recall_method_b(pred_b["retrieved_docs"], gt_strs, k)
            recalls_a[k].append(ra)
            recalls_b[k].append(rb)
            ra_row[k] = ra
            rb_row[k] = rb
            if abs(ra - rb) > 1e-9:
                row_diff = True
        if row_diff:
            diff_queries.append((pred_a, pred_b, gt))

    # ── 汇总 Recall@K ──────────────────────────────────────────────────────────
    w = max(len(label_a), len(label_b), 6)
    print("\n" + "=" * 80)
    print("📊 总体 Recall@K 对比（所有查询均值）")
    print("=" * 80)
    header = (f"{'K':>4}  {label_a:>{w+9}}  {label_b:>{w+9}}  {'差值(A-B)':>10}")
    print(header)
    print("─" * (4 + 2 + w + 9 + 2 + w + 9 + 2 + 12))
    for k in sorted(k_values):
        la, lb = recalls_a[k], recalls_b[k]
        if not la:
            continue
        avg_a = sum(la) / len(la)
        avg_b = sum(lb) / len(lb)
        diff  = avg_a - avg_b
        flag  = " ← 有差异" if abs(diff) > 1e-9 else ""
        print(f"  @{k:<3}  {avg_a:.4f} ({avg_a*100:5.1f}%)  "
              f"{avg_b:.4f} ({avg_b*100:5.1f}%)  "
              f"{diff:+.4f}{flag}")

    total = len(aligned)
    diff_count = len(diff_queries)
    print(f"\n{'─'*80}")
    print(f"总查询数:  {total}")
    print(f"结果一致:  {total - diff_count}  ({(total-diff_count)/total*100:.1f}%)")
    print(f"结果不同:  {diff_count}  ({diff_count/total*100:.1f}%)")

    print(f"\n{'─'*80}")
    print("📈 各 K 下 Recall 分布对比")
    print(f"{'─'*80}")
    for k in sorted(k_values):
        la, lb = recalls_a[k], recalls_b[k]
        if not la:
            continue

        def dist(lst):
            full    = sum(1 for r in lst if r >= 1.0)
            partial = sum(1 for r in lst if 0 < r < 1.0)
            zero    = sum(1 for r in lst if r == 0.0)
            return full, partial, zero

        fa, pa, za = dist(la)
        fb, pb, zb = dist(lb)
        print(f"  @{k}: {label_a}  full={fa:4d} partial={pa:4d} zero={za:4d}"
              f"  |  {label_b}  full={fb:4d} partial={pb:4d} zero={zb:4d}")

    if verbose and diff_queries:
        print("\n" + "=" * 120)
        print(f"📝 逐条差异详情（共 {diff_count} 条，仅展示结果不一致的查询）")
        print("=" * 120)
        for i, (pred_a, pred_b, gt) in enumerate(diff_queries, 1):
            print_query_detail_compare(
                i, diff_count, pred_a, pred_b, gt,
                k_values, max_k, label_a, label_b
            )
    elif verbose:
        print("\n✅ 所有查询两个文件结果完全一致，无差异。")

    # ── 0 召回问题列表 ────────────────────────────────────────────────────────
    check_ks = sorted({5, max_k} & set(k_values))   # 只统计 k_values 中存在的 K

    def zero_tuple_list(k):
        return [(pa, pb, gt) for pa, pb, gt in aligned
                if recall_method_b(pa["retrieved_docs"], gt["gt_strings"], k) == 0.0
                and recall_method_b(pb["retrieved_docs"], gt["gt_strings"], k) == 0.0]

    def zero_single_list(k, use_a=True):
        return [(pa, pb, gt) for pa, pb, gt in aligned
                if recall_method_b(
                    (pa if use_a else pb)["retrieved_docs"], gt["gt_strings"], k
                ) == 0.0]

    for ck in check_ks:
        za = zero_single_list(ck, use_a=True)
        zb = zero_single_list(ck, use_a=False)
        zb_both = zero_tuple_list(ck)

        print(f"\n{'─'*80}")
        print(f"🚫 0召回问题统计（@{ck} 下召回率仍为 0）")
        print(f"{'─'*80}")
        print(f"  {label_a}: {len(za):4d} 条（{len(za)/total*100:.1f}%）")
        print(f"  {label_b}: {len(zb):4d} 条（{len(zb)/total*100:.1f}%）")
        print(f"  两者均为0: {len(zb_both):4d} 条（{len(zb_both)/total*100:.1f}%）")

        if za:
            print(f"\n  ── {label_a} 的0召回问题（@{ck}）──")
            for i, (pa, pb, gt) in enumerate(za, 1):
                question = gt["question"] or pa["question"]
                gt_titles = ", ".join(doc["title"] for doc in gt["gt_docs"])
                print(f"  {i:3d}. 问题: {question}")
                print(f"       GT 标题: {gt_titles}")

        if zb:
            print(f"\n  ── {label_b} 的0召回问题（@{ck}）──")
            for i, (pa, pb, gt) in enumerate(zb, 1):
                question = gt["question"] or pb["question"]
                gt_titles = ", ".join(doc["title"] for doc in gt["gt_docs"])
                print(f"  {i:3d}. 问题: {question}")
                print(f"       GT 标题: {gt_titles}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="用方法B（原始字符串精确匹配）评估并对比检索结果的 Recall@K",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--predictions", type=str, nargs="+", required=True,
        metavar="FILE",
        help="检索结果 JSON 文件，传入 1 个时单独评估，传入 2 个时互相对比",
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", default=None,
        metavar="LABEL",
        help="对应每个 --predictions 文件的显示标签（可选，默认取文件名）",
    )

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset", type=str,
                               help="数据集 JSON 完整路径")
    dataset_group.add_argument("--dataset-name", type=str,
                               help="数据集名称，自动映射到 pipeline/evaluation/dataset/<name>.json")

    parser.add_argument("--k-values", type=str, default="1,2,3,5,10",
                        help="K 值列表，逗号分隔（默认: 1,2,3,5,10）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只处理前 N 条，方便快速调试")
    parser.add_argument("--verbose", action="store_true",
                        help="输出逐条详情（单文件）或差异详情（双文件）")

    args = parser.parse_args()

    if len(args.predictions) > 2:
        parser.error("--predictions 最多支持 2 个文件")

    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    # 确定数据集路径
    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_path = f"./pipeline/evaluation/dataset/{args.dataset_name}.json"

    # 确定标签
    def default_label(path: str) -> str:
        return Path(path).stem

    if args.labels:
        labels = args.labels + [default_label(p) for p in args.predictions[len(args.labels):]]
    else:
        labels = [default_label(p) for p in args.predictions]

    print("=" * 80)
    print("Recall@K 评估（方法B：原始字符串精确匹配）")
    print("=" * 80)
    for path, label in zip(args.predictions, labels):
        print(f"  [{label}]: {path}")
    print(f"  数据集:   {dataset_path}")
    print(f"  K 值:     {k_values}")
    print()
    print("  方法 B：原始字符串精确匹配")
    print('    → 将 GT 构建为 "title\\ncontent" 原始字符串')
    print("    → 直接用 == 比较，不做任何标准化")
    print("=" * 80)
    print()

    # 加载数据集（共享）
    ground_truth = load_ground_truth(dataset_path)

    # 加载并对齐每个预测文件
    all_paired = []
    for path in args.predictions:
        preds = load_predictions(path)
        if args.limit:
            preds = preds[:args.limit]
        paired = align_by_question(
            preds,
            ground_truth[:args.limit] if args.limit else ground_truth,
        )
        all_paired.append(paired)

    if args.limit:
        print(f"✓ 限制处理前 {args.limit} 条\n")

    if len(all_paired) == 1:
        run_single(all_paired[0], k_values, labels[0], args.verbose)
    else:
        run_compare(
            all_paired[0], all_paired[1],
            k_values,
            labels[0], labels[1],
            args.verbose,
        )


if __name__ == "__main__":
    main()
