# 数据集加载器 (DatasetLoader)

## 概述

`DatasetLoader` 类用于加载评估数据集，包括 corpus、问题数据集和评估标签（gold_docs、gold_answers）。实现参考了 HippoRAG 项目的数据加载方式。

## 支持的数据集

- `musique` - MuSiQue 多跳问答数据集
- `hotpotqa` - HotpotQA 数据集
- `hotpotqa_train` - HotpotQA 训练集
- `2wikimultihopqa` - 2WikiMultihopQA 数据集

## 快速开始

### 方式1：使用 DatasetLoader 类

```python
from dataflow.evaluation.utils import DatasetLoader

# 创建加载器
loader = DatasetLoader('musique')

# 获取数据
docs = loader.get_docs()                    # 格式化的文档列表 ["title\ntext"]
questions = loader.get_questions()          # 问题列表
gold_answers = loader.get_gold_answers()    # 标准答案列表（集合）
gold_docs = loader.get_gold_docs()          # 支持文档列表

# 获取统计信息
stats = loader.get_stats()
print(stats)
```

### 方式2：使用便捷函数

```python
from dataflow.evaluation.utils import load_dataset

# 一次性加载所有常用数据
docs, questions, gold_answers, gold_docs = load_dataset('musique')

print(f"加载了 {len(docs)} 个文档，{len(questions)} 个问题")
```

### 方式3：使用独立辅助函数

```python
from dataflow.evaluation.utils import get_gold_answers, get_gold_docs
import json

# 直接从样本列表提取
samples = json.load(open('dataset/musique.json'))
gold_answers = get_gold_answers(samples)
gold_docs = get_gold_docs(samples, dataset_name='musique')
```

## API 参考

### DatasetLoader 类

#### 初始化

```python
loader = DatasetLoader(
    dataset_name: str,              # 数据集名称
    dataset_dir: Optional[str] = None  # 数据集目录（默认：pipeline/evaluation/dataset）
)
```

#### 主要方法

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `load_corpus()` | `List[Dict[str, str]]` | 加载原始corpus数据 |
| `load_samples()` | `List[Dict[str, Any]]` | 加载问题样本数据 |
| `get_docs()` | `List[str]` | 获取格式化文档列表（"title\ntext"） |
| `get_questions()` | `List[str]` | 获取问题列表 |
| `get_gold_answers()` | `List[Set[str]]` | 提取标准答案（含别名） |
| `get_gold_docs()` | `Optional[List[List[str]]]` | 提取支持文档 |
| `get_stats()` | `Dict[str, Any]` | 获取数据集统计信息 |
| `load_all()` | `Dict[str, Any]` | 一次性加载所有数据 |

### 便捷函数

```python
def load_dataset(
    dataset_name: str,
    dataset_dir: Optional[str] = None
) -> Tuple[List[str], List[str], List[Set[str]], Optional[List[List[str]]]]:
    """返回 (docs, questions, gold_answers, gold_docs)"""
```

### 独立辅助函数

```python
def get_gold_answers(samples: List[Dict[str, Any]]) -> List[Set[str]]:
    """从样本列表提取标准答案"""

def get_gold_docs(samples: List[Dict[str, Any]], dataset_name: str = None) -> List[List[str]]:
    """从样本列表提取支持文档"""
```

## 数据格式说明

### Corpus 格式

```json
[
  {
    "title": "文档标题",
    "text": "文档内容..."
  }
]
```

### 样本格式

#### MuSiQue

```json
{
  "id": "样本ID",
  "question": "问题文本",
  "answer": "答案",
  "answer_aliases": ["别名1", "别名2"],
  "paragraphs": [
    {
      "idx": 0,
      "title": "段落标题",
      "paragraph_text": "段落内容",
      "is_supporting": true
    }
  ]
}
```

#### HotpotQA

```json
{
  "_id": "样本ID",
  "question": "问题文本",
  "answer": "答案",
  "supporting_facts": [["文档标题", 句子索引]],
  "context": [
    ["文档标题", ["句子1", "句子2", ...]]
  ]
}
```

## 数据集统计

运行示例可查看各数据集统计信息：

```bash
python pipeline/evaluation/examples/dataset_loader_example.py
```

### 典型统计

| 数据集 | Corpus文档数 | 问题数 | 平均支持文档数/题 |
|--------|-------------|--------|------------------|
| musique | 11,656 | 1,000 | 2.65 |
| hotpotqa | 9,811 | 1,000 | 2.00 |
| 2wikimultihopqa | 6,119 | 1,000 | 2.47 |

## 完整示例

参考 `pipeline/evaluation/examples/dataset_loader_example.py` 获取更多使用示例。

## 与 HippoRAG 的兼容性

此实现与 HippoRAG 的数据加载方式保持一致：

1. ✅ 相同的 corpus 加载格式 `"title\ntext"`
2. ✅ 相同的 gold_answers 提取逻辑（支持别名）
3. ✅ 相同的 gold_docs 提取逻辑（支持多种数据集格式）
4. ✅ 相同的数据集目录结构

可以直接替换 HippoRAG 中的数据加载代码：

```python
# HippoRAG 原始代码
corpus = json.load(open(f"reproduce/dataset/{dataset_name}_corpus.json"))
docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
samples = json.load(open(f"reproduce/dataset/{dataset_name}.json"))
gold_answers = get_gold_answers(samples)
gold_docs = get_gold_docs(samples, dataset_name)

# 使用 DatasetLoader（等价）
loader = DatasetLoader(dataset_name)
docs = loader.get_docs()
gold_answers = loader.get_gold_answers()
gold_docs = loader.get_gold_docs()
```

## 日志

加载器使用项目的统一日志系统：

```python
from dataflow.utils import get_logger
logger = get_logger("evaluation.utils.load_utils")
```

日志示例：
```
dataflow.evaluation.utils.load_utils - INFO - Loading corpus from .../musique_corpus.json
dataflow.evaluation.utils.load_utils - INFO - Loaded 11656 documents from corpus
dataflow.evaluation.utils.load_utils - INFO - Extracted 1000 gold answers
```

## 错误处理

```python
loader = DatasetLoader('musique')

# 文件不存在时抛出 FileNotFoundError
try:
    docs = loader.get_docs()
except FileNotFoundError as e:
    print(f"数据集文件未找到: {e}")

# gold_docs 提取失败时返回 None
gold_docs = loader.get_gold_docs()
if gold_docs is None:
    print("该数据集不支持 gold_docs 评估")
```

## 扩展自定义数据集

只需遵循以下文件命名规范：

```
dataset_dir/
  ├── my_dataset_corpus.json    # corpus 文件
  └── my_dataset.json           # 样本文件
```

然后加载：

```python
loader = DatasetLoader('my_dataset', dataset_dir='/path/to/dataset_dir')
```
