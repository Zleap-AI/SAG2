"""
数据集加载工具模块

提供数据集加载、corpus加载和评估标签提取功能
"""

import json
import shutil
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path

from pipeline.utils import get_logger

logger = get_logger("evaluation.utils.load_utils")


class DatasetLoader:
    """
    数据集加载类

    支持加载corpus、问题数据集、提取评估标签（gold_docs、gold_answers）
    支持的数据集：musique, hotpotqa, 2wikimultihopqa, sample
    """

    SUPPORTED_DATASETS = ['musique', 'hotpotqa', '2wikimultihopqa', 'sample', 'test_hotpotqa']

    def __init__(self, dataset_name: str, dataset_dir: Optional[str] = None):
        """
        初始化数据集加载器

        Args:
            dataset_name: 数据集名称 (musique, hotpotqa, 2wikimultihopqa, sample)
            dataset_dir: 数据集目录路径，默认为 pipeline/evaluation/dataset
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            logger.warning(
                f"Dataset '{dataset_name}' is not in supported list {self.SUPPORTED_DATASETS}, "
                f"but will attempt to load it."
            )

        self.dataset_name = dataset_name

        # 默认数据集目录
        if dataset_dir is None:
            current_file = Path(__file__)
            dataset_dir = current_file.parent.parent / "dataset"

        self.dataset_dir = Path(dataset_dir)

        # 数据集文件路径
        self.corpus_path = self.dataset_dir / f"{dataset_name}_corpus.json"
        self.samples_path = self.dataset_dir / f"{dataset_name}.json"

        # 缓存数据
        self._corpus = None
        self._samples = None
        self._docs = None
        self._questions = None
        self._gold_answers = None
        self._gold_docs = None

        logger.info(f"Initialized DatasetLoader for '{dataset_name}' with directory: {self.dataset_dir}")

    def load_corpus(self, force_reload: bool = False) -> List[Dict[str, str]]:
        """
        加载corpus数据

        Args:
            force_reload: 是否强制重新加载

        Returns:
            corpus列表，每个元素包含 'title' 和 'text' 字段
        """
        if self._corpus is not None and not force_reload:
            return self._corpus

        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        logger.info(f"Loading corpus from {self.corpus_path}")

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            self._corpus = json.load(f)

        logger.info(f"Loaded {len(self._corpus)} documents from corpus")
        return self._corpus

    def load_samples(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """
        加载问题样本数据

        Args:
            force_reload: 是否强制重新加载

        Returns:
            样本列表
        """
        if self._samples is not None and not force_reload:
            return self._samples

        if not self.samples_path.exists():
            raise FileNotFoundError(f"Samples file not found: {self.samples_path}")

        logger.info(f"Loading samples from {self.samples_path}")

        with open(self.samples_path, "r", encoding="utf-8") as f:
            self._samples = json.load(f)

        logger.info(f"Loaded {len(self._samples)} samples")
        return self._samples

    def get_docs(self, force_reload: bool = False) -> List[str]:
        """
        获取格式化的文档列表（title + text）

        Args:
            force_reload: 是否强制重新加载

        Returns:
            文档文本列表，格式为 "title\\ntext"
        """
        if self._docs is not None and not force_reload:
            return self._docs

        corpus = self.load_corpus(force_reload)
        self._docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

        return self._docs

    def get_questions(self, force_reload: bool = False) -> List[str]:
        """
        获取问题列表

        Args:
            force_reload: 是否强制重新加载

        Returns:
            问题文本列表
        """
        if self._questions is not None and not force_reload:
            return self._questions

        samples = self.load_samples(force_reload)
        self._questions = [s['question'] for s in samples]

        return self._questions

    def get_gold_answers(self, force_reload: bool = False) -> List[Set[str]]:
        """
        提取gold answers（参考 HippoRAG 实现）

        Args:
            force_reload: 是否强制重新加载

        Returns:
            gold answers列表，每个元素是一个集���（包含答案及其别名）
        """
        if self._gold_answers is not None and not force_reload:
            return self._gold_answers

        samples = self.load_samples(force_reload)
        gold_answers = []

        for sample_idx, sample in enumerate(samples):
            gold_ans = None

            # 尝试不同的答案字段
            if 'answer' in sample or 'gold_ans' in sample:
                gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
            elif 'reference' in sample:
                gold_ans = sample['reference']
            elif 'obj' in sample:
                # 合并多个可能的答案字段
                gold_ans = set(
                    [sample['obj']] +
                    [sample.get('possible_answers', '')] +
                    [sample.get('o_wiki_title', '')] +
                    [sample.get('o_aliases', '')]
                )
                gold_ans = list(gold_ans)

            if gold_ans is None:
                logger.warning(f"Sample {sample_idx} has no answer field, skipping")
                gold_answers.append(set())
                continue

            # 确保答案是列表格式
            if isinstance(gold_ans, str):
                gold_ans = [gold_ans]

            assert isinstance(gold_ans, list), f"Gold answer should be a list, got {type(gold_ans)}"

            # 转换为集合
            gold_ans_set = set(gold_ans)

            # 添加答案别名
            if 'answer_aliases' in sample:
                gold_ans_set.update(sample['answer_aliases'])

            gold_answers.append(gold_ans_set)

        self._gold_answers = gold_answers
        logger.info(f"Extracted {len(gold_answers)} gold answers")

        return gold_answers

    def get_gold_docs(self, force_reload: bool = False) -> Optional[List[List[str]]]:
        """
        提取gold documents（兼容所有数据集格式）

        Args:
            force_reload: 是否强制重新加载

        Returns:
            gold docs列表，每个元素是支持文档列表；如果数据集不支持则返回None
        """
        return self.get_gold_docs_for_recall(force_reload=force_reload)

    def get_gold_docs_for_recall(self, force_reload: bool = True, limit: Optional[int] = None) -> Optional[List[List[str]]]:
        """
        获取用于召回率评估的gold docs（带标题和内容）

        Args:
            force_reload: 是否强制重新加载
            limit: 限制返回的问题数量，None表示返回全部

        Returns:
            gold docs列表，每个元素是文档列表，文档格式为 "title\ncontent" 字符串
        """
        # 重新提取
        samples = self.load_samples(force_reload)

        try:
            gold_docs_list = []

            for idx, sample in enumerate(samples):
                # 应用 limit
                if limit is not None and idx >= limit:
                    break

                docs = []

                # hotpotqa, 2wikimultihopqa: supporting_facts + context
                if self.dataset_name in ["hotpotqa", "2wikimultihopqa", "test_sample", "test_hotpotqa"]:
                    gold_title = set([item[0] for item in sample['supporting_facts']])
                    for item in sample['context']:
                        if item[0] in gold_title:
                            content = ''.join(item[1]) if (self.dataset_name in ['hotpotqa', 'test_hotpotqa']) else ' '.join(item[1])
                            docs.append(f"{item[0]}\n{content}")

                # musique: paragraphs
                elif self.dataset_name == "musique":
                    for item in sample['paragraphs']:
                        if item.get('is_supporting', False):
                            content = item.get('paragraph_text', '')
                            docs.append(f"{item['title']}\n{content}")

                gold_docs_list.append(docs)

            return gold_docs_list

        except Exception as e:
            logger.error(f"Failed to extract gold docs for recall: {e}")
            return None

    def load_all(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        加载所有数据

        Args:
            force_reload: 是否强制重新加载

        Returns:
            包含所有数据的字典
        """
        questions = self.get_questions(force_reload)
        samples = self.load_samples(force_reload)

        # 提取 paragraphs（如果存在）
        all_paragraphs = []
        for sample in samples:
            all_paragraphs.append(sample.get('paragraphs', []))

        return {
            'corpus': self.load_corpus(force_reload),
            'samples': samples,
            'docs': self.get_docs(force_reload),
            'questions': questions,
            'answers': self.get_gold_answers(force_reload),  # 添加 answers 字段
            'paragraphs': all_paragraphs,  # 添加 paragraphs 字段
            'gold_answers': self.get_gold_answers(force_reload),
            'gold_docs': self.get_gold_docs(force_reload),
            'total_questions': len(questions),  # 添加总问题数
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        获取数据集统计信息

        Returns:
            统计信息字典
        """
        corpus = self.load_corpus()
        samples = self.load_samples()
        questions = self.get_questions()
        gold_answers = self.get_gold_answers()
        gold_docs = self.get_gold_docs()

        stats = {
            'dataset_name': self.dataset_name,
            'num_corpus_docs': len(corpus),
            'num_samples': len(samples),
            'num_questions': len(questions),
            'num_gold_answers': len(gold_answers),
            'has_gold_docs': gold_docs is not None,
        }

        if gold_docs is not None:
            stats['num_gold_docs'] = len(gold_docs)
            stats['avg_supporting_docs_per_question'] = (
                sum(len(docs) for docs in gold_docs) / len(gold_docs)
                if gold_docs else 0
            )

        return stats

    def save_as_markdown(
        self,
        output_dir: Optional[str] = None,
        chunks_per_file: int = 500,
        force_regenerate: bool = False
    ) -> Dict[str, Any]:
        """
        将数据集的 **corpus**（所有文档）保存为 markdown 格式

        从 corpus 中加载所有文档，每个文档转换为 markdown 格式：
        - title 添加 # 符号作为标题
        - text 作为正文内容
        每 chunks_per_file 个文档组成一个 markdown 文件

        Args:
            output_dir: 输出目录，默认为 pipeline/evaluation/markdown_datasets/{dataset_name}/
            chunks_per_file: 每个 markdown 文件包含的文档数量
            force_regenerate: 是否强制重新生成，如果为 True 则覆盖已存在的文件

        Returns:
            包含输出目录和统计信息的字典:
            {
                'output_dir': Path,
                'stats': {
                    'total_chunks': int,        # corpus 中的总文档数
                    'num_files': int,           # 生成的 markdown 文件数
                    'chunks_per_file': int,     # 每个文件的文档数
                    'last_file_chunks': int or None  # 最后一个文件的文档数
                }
            }
        """
        # 确定输出目录
        if output_dir is None:
            current_file = Path(__file__)
            output_dir = current_file.parent.parent / "markdown_datasets" / self.dataset_name

        output_path = Path(output_dir)

        # 删除旧的输出目录（包括所有文件）
        if output_path.exists():
            logger.info(f"Removing old output directory: {output_path}")
            shutil.rmtree(output_path)
            logger.info(f"Old directory removed successfully")

        # 创建新的输出目录
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving dataset '{self.dataset_name}' as markdown to {output_path}")

        # 加载 corpus（包含所有文档）
        corpus = self.load_corpus()

        # 收集所有文档（chunks）
        all_chunks = []
        chunk_count = 0

        for doc_idx, doc in enumerate(corpus):
            title = doc.get('title', f"Document {doc_idx}")
            text = doc.get('text', '')

            all_chunks.append({
                'title': title,
                'text': text,
                'doc_idx': doc_idx
            })
            chunk_count += 1

        logger.info(f"Collected {chunk_count} chunks from corpus")

        # 将 chunks 分块并保存为 markdown 文件
        file_count = 0
        for i in range(0, len(all_chunks), chunks_per_file):
            chunk_batch = all_chunks[i:i + chunks_per_file]
            file_count += 1

            # 构造 markdown 内容（只包含标题和文本）
            markdown_content = []

            for chunk_idx, chunk in enumerate(chunk_batch):
                # 添加标题
                markdown_content.append(f"# {chunk['title']}")
                # 添加文本内容
                markdown_content.append(chunk['text'])
                markdown_content.append("")
                markdown_content.append("")

            # 保存文件
            output_file = output_path / f"{self.dataset_name}_part{file_count}.md"

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(markdown_content))

            logger.info(f"Saved markdown file: {output_file} ({len(chunk_batch)} chunks)")

        logger.info(f"Total saved {file_count} markdown files to {output_path}")

        # 收集统计信息
        stats = {
            'total_chunks': chunk_count,
            'num_files': file_count,
            'chunks_per_file': chunks_per_file,
            'last_file_chunks': None
        }

        if file_count > 0:
            # 检查最后一个文件的实际 chunk 数量
            files = sorted(output_path.glob("*.md"), key=lambda p: int(p.stem.split('_part')[-1]))
            if files:
                last_file = files[-1]
                with open(last_file, 'r', encoding='utf-8') as f:
                    last_chunks = sum(1 for line in f if line.strip().startswith('#') and line.strip() != '#')
                stats['last_file_chunks'] = last_chunks

        # 返回输出目录和统计信息
        return {
            'output_dir': output_path,
            'stats': stats
        }


def load_dataset(
    dataset_name: str,
    dataset_dir: Optional[str] = None
) -> Tuple[List[str], List[str], List[Set[str]], Optional[List[List[str]]]]:
    """
    便捷函数：加载数据集并返回常用数据

    Args:
        dataset_name: 数据集名称
        dataset_dir: 数据集目录路径

    Returns:
        (docs, questions, gold_answers, gold_docs)
    """
    loader = DatasetLoader(dataset_name, dataset_dir)

    docs = loader.get_docs()
    questions = loader.get_questions()
    gold_answers = loader.get_gold_answers()
    gold_docs = loader.get_gold_docs()

    logger.info(f"Loaded dataset '{dataset_name}': "
                f"{len(docs)} docs, {len(questions)} questions")

    return docs, questions, gold_answers, gold_docs


# 为了兼容性，提供独立的辅助函数
def get_gold_answers(samples: List[Dict[str, Any]]) -> List[Set[str]]:
    """
    从样本列表中提取gold answers（独立函数）

    Args:
        samples: 样本列表

    Returns:
        gold answers列表
    """
    gold_answers = []

    for sample_idx, sample in enumerate(samples):
        gold_ans = None

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] +
                [sample.get('possible_answers', '')] +
                [sample.get('o_wiki_title', '')] +
                [sample.get('o_aliases', '')]
            )
            gold_ans = list(gold_ans)

        if gold_ans is None:
            logger.warning(f"Sample {sample_idx} has no answer field")
            gold_answers.append(set())
            continue

        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]

        assert isinstance(gold_ans, list)

        gold_ans_set = set(gold_ans)

        if 'answer_aliases' in sample:
            gold_ans_set.update(sample['answer_aliases'])

        gold_answers.append(gold_ans_set)

    return gold_answers
