# Multi-hop QA Dataset Documentation

本文档介绍了三个经典的多跳问答（Multi-hop QA）数据集：HotPotQA、2WikiMultiHopQA 和 MuSiQue。这些数据集是多跳推理和检索增强生成（RAG）研究中的标准评测基准。

## 目录

- [数据集概述](#数据集概述)
- [HotPotQA](#hotpotqa)
- [2WikiMultiHopQA](#2wikimultihopqa)
- [MuSiQue](#musique)
- [统一数据格式](#统一数据格式)
- [问题类型说明](#问题类型说明)

---

## 数据集概述

### 训练集、验证集和测试集

在机器学习中，数据集通常分为三种类型：

- **训练集（Train Dataset）**：用于训练模型，帮助模型学习数据特征。包含大量样本，是数据集中最大的部分。
- **验证集（Dev Dataset）**：用于调整模型超参数和模型选择，评估模型泛化能力。在训练过程中多次使用，防止过拟合。
- **测试集（Test Dataset）**：用于评估最终模型性能，提供模型在真实应用场景中的表现指标。与训练集和验证集完全独立。

---

## HotPotQA

### 基本信息

- **论文**：Yang et al. - 2018 - HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering
- **论文链接**：[arXiv:1809.09600](https://arxiv.org/abs/1809.09600)
- **项目地址**：[HotpotQA Homepage](https://hotpotqa.github.io/)
- **发布机构**：卡内基梅隆大学（2018年）

### 构建思路

HotPotQA 是一个用于多样化、可解释的多跳问题解答的数据集，旨在评估模型在需要跨多个文档进行推理的问答任务中的表现。

### 数据格式

#### 训练集格式（hotpot_train_v1.1.json）

```json
{
    "supporting_facts": [
        ["str(title)", "int(sent_id)"]
    ],
    "level": "easy" | "hard" | "medium",
    "question": "str",
    "context": [
        [
            "str(Title)",
            [
                "str(Sent)",
                "str(Sent)"
            ]
        ]
    ],
    "answer": "str",
    "_id": "str",
    "type": "bridge" | "comparison"
}
```

**字段说明**：

- `supporting_facts`：支持文档的标题和对应句子的序号（第几句）
- `level`：问题难度级别（easy/hard/medium）
- `question`：问题文本
- `context`：包含约10个文档，每个文档由标题和句子列表组成
- `answer`：问题的答案
- `_id`：唯一标识符
- `type`：问题类型
  - `bridge`：需要跨越多个上下文信息来连接事实并推导答案
  - `comparison`：需要对比多个事实来得出答案

#### 验证集格式

- **hotpot_dev_distractor_v1.json**：包含2个关键文档和8个干扰文档
- **hotpot_dev_fullwiki_v1.json**：更具挑战性，context 字段不一定包含关键文档，要求模型从全部维基百科的第一段中检索并找到关键信息

#### 测试集格式（hotpot_test_fullwiki_v1.json）

```json
[
    {
        "_id": "str",
        "question": "str",
        "context": [
            [
                "str(Title)",
                [
                    "str(Sent)",
                    "str(Sent)"
                ]
            ]
        ]
    }
]
```

测试集只包含问题和上下文，答案不公开。需要在 [HotpotQA Homepage](https://hotpotqa.github.io/) 上传模型输出结果获得评估分数。

#### 开放域问答

可在项目主页下载作者团队处理过的 Wiki 离线语料库。

### 示例

```json
{
    "supporting_facts": [
        ["Allie Goertz", 0],
        ["Allie Goertz", 1],
        ["Allie Goertz", 2],
        ["Milhouse Van Houten", 0]
    ],
    "level": "hard",
    "question": "Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?",
    "context": [
        [
            "Allie Goertz",
            [
                "Allison Beth \"Allie\" Goertz (born March 2, 1991) is an American musician.",
                "Goertz is known for her satirical songs based on various pop culture topics.",
                "Her videos are posted on YouTube under the name of Cossbysweater.",
                "Subjects of her songs have included the film \"The Room\", the character Milhouse from the television show \"The Simpsons\", and the game Dungeons & Dragons."
            ]
        ],
        [
            "Milhouse Van Houten",
            [
                "Milhouse Mussolini van Houten is a fictional character featured in the animated television series \"The Simpsons\", voiced by Pamela Hayden, and created by Matt Groening who named the character after President Richard Nixon's middle name.",
                "Later in the series, it is revealed that Milhouse's middle name is \"Mussolini.\""
            ]
        ]
    ],
    "answer": "President Richard Nixon",
    "_id": "5a8d7341554299441c6b9fe5",
    "type": "bridge"
}
```

---

## 2WikiMultiHopQA

### 基本信息

- **论文**：Ho et al. - 2020 - Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps
- **论文链接**：[ACL Anthology](https://aclanthology.org/2020.coling-main.580/)
- **项目地址**：[Alab-NII/2wikimultihop](https://github.com/Alab-NII/2wikimultihop)

### 构建思路

研究表明 HotpotQA 中的许多示例不需要多跳推理即可解决。为此，作者在每个样本中引入了新信息：包含全面而简洁的证据来解释预测。

本数据集中的证据信息是一组三元组，每个三元组都是从 Wikidata 获得的结构化数据（主题实体、属性、对象实体），可用于解释预测并测试模型的推理能力。

### 数据格式

#### 训练集/验证集（train.json、dev.json）

```json
{
    "_id": "str",
    "type": "compositional" | "inference" | "bridge_comparison" | "comparison",
    "question": "str",
    "context": [
        [
            "str(Title)",
            [
                "str(Sent)",
                "str(Sent)"
            ]
        ]
    ],
    "supporting_facts": [
        ["str(title)", "int(sent_id)"]
    ],
    "evidences": [
        ["str(subject entity)", "str(relation)", "str(object entity)"]
    ],
    "answer": "str"
}
```

**字段说明**：

- `_id`：每个样本的唯一标识
- `type`：问题类型（compositional/inference/bridge_comparison/comparison）
- `question`：问题文本
- `context`：文档列表，每个文档包含标题和句子列表
- `supporting_facts`：支持文档的标题和句子索引（从0开始）
- `evidences`：三元组列表，每个三元组包含 [主体实体, 关系, 客体实体]。有几组 supporting_facts 就有几组 evidences
- `answer`：问题答案

#### 测试集

与训练集和验证集格式相同，但 `answer`、`supporting_facts` 和 `evidences` 留空。

进行评测需要联系作者，格式要求见 GitHub。

#### 开放域问答

可在项目地址下载作者团队处理过的 Wiki 离线语料库：para_with_hyperlink.zip

### 示例

```json
{
    "_id": "13f5ad2c088c11ebbd6fac1f6bf848b6",
    "type": "bridge_comparison",
    "question": "Are director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?",
    "context": [
        [
            "Move (1970 film)",
            [
                "Move is a 1970 American comedy film starring Elliott Gould, Paula Prentiss and Geneviève Waïte, and directed by Stuart Rosenberg.",
                "The screenplay was written by Joel Lieber and Stanley Hart, adapted from a novel by Lieber."
            ]
        ],
        [
            "Stuart Rosenberg",
            [
                "Stuart Rosenberg (August 11, 1927 – March 15, 2007) was an American film and television director whose motion pictures include \"Cool Hand Luke\" (1967), \"Voyage of the Damned\" (1976), \"The Amityville Horror\" (1979), and \"The Pope of Greenwich Village\" (1984).",
                "He was noted for his work with actor Paul Newman."
            ]
        ],
        [
            "Méditerranée (1963 film)",
            [
                "Méditerranée is a 1963 French experimental film directed by Jean-Daniel Pollet with assistance from Volker Schlöndorff."
            ]
        ],
        [
            "Jean-Daniel Pollet",
            [
                "Jean-Daniel Pollet (1936–2004) was a French film director and screenwriter who was most active in the 1960s and 1970s."
            ]
        ]
    ],
    "supporting_facts": [
        ["Move (1970 film)", 0],
        ["Méditerranée (1963 film)", 0],
        ["Stuart Rosenberg", 0],
        ["Jean-Daniel Pollet", 0]
    ],
    "evidences": [
        ["Move (1970 film)", "director", "Stuart Rosenberg"],
        ["Méditerranée (1963 film)", "director", "Jean-Daniel Pollet"],
        ["Stuart Rosenberg", "country of citizenship", "American"],
        ["Jean-Daniel Pollet", "country of citizenship", "French"]
    ],
    "answer": "no"
}
```

---

## MuSiQue

### 基本信息

- **论文**：Trivedi et al. - 2022 - ♫ MuSiQue: Multihop Questions via Single-hop Question Composition
- **论文链接**：[TACL 2022](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00475/111040)
- **项目地址**：[StonyBrookNLP/musique](https://github.com/StonyBrookNLP/musique)

### 构建思路

MuSiQue 引入了一种自下而上的过程，通过仔细选择和组合从现有数据集中获得的单跳问题，构建具有挑战性的多跳阅读理解问答数据集。

**关键思想**：

1. 从大量单跳问题中组合多跳问题，系统地探索广泛的候选多跳问题空间
2. 应用严格的筛选标准，确保没有子问题能够在未找到其连接的前一个子问题答案的情况下被回答
3. 在每个单跳问题的层面上减少训练-测试泄漏，减轻简单记忆技巧的影响
4. 添加难以识别的干扰上下文
5. 在子问题层面上创建不可回答的多跳问题

**数据集版本**：

- **MuSiQue-Ans**：约25,000个2-4跳的问题，具有六种不同的组合结构
- **MuSiQue-Full**：约50,000个多跳问题，包含可回答与不可回答问题的对比对，挑战性更高

### 数据格式

#### 训练集/验证集（train.json、dev.json）

```json
{
    "id": "nhop_indx(str)",
    "paragraphs": [
        {
            "idx": "int",
            "title": "str",
            "paragraph_text": "str",
            "is_supporting": true | false
        }
    ],
    "question": "str",
    "question_decomposition": [
        {
            "id": "int",
            "question": "str",
            "answer": "str",
            "paragraph_support_idx": "int"
        }
    ],
    "answer": "str",
    "answer_aliases": [
        "str"
    ],
    "answerable": true | false
}
```

**字段说明**：

- `id`：唯一标识符，`nhop` 表示该问题有几跳
- `paragraphs`：段落列表
  - `idx`：段落索引编号
  - `title`：段落标题
  - `paragraph_text`：段落内容
  - `is_supporting`：该段落是否支持回答问题。在 `-ans` 数据中一定存在 true 字段，在 `-full` 数据中则不一定
- `question`：问题文本
- `question_decomposition`：问题分解列表（注意：这里的问题不是简单子问题，而是涵盖推理步骤的子问题拆解，会有一些奇怪的代词）
  - `id`：分解后问题的编号
  - `question`：分解问题的内容
  - `answer`：分解问题的答案
  - `paragraph_support_idx`：支持该分解问题答案的段落索引
- `answer`：最终答案
- `answer_aliases`：答案的别名或同义词列表
- `answerable`：问题是否可以被回答。在 `-ans` 数据中都为 true

#### 测试集

测试集只保留 `id`、`paragraphs`（去掉 `is_supporting`）和 `question` 字段。提交方式见 GitHub。

### 示例

```json
{
    "id": "2hop__28482_46077",
    "paragraphs": [
        {
            "idx": 4,
            "title": "Baltic Sea",
            "paragraph_text": "Since May 2004, with the accession of the Baltic states and Poland, the Baltic Sea has been almost entirely surrounded by countries of the European Union (EU). The only remaining non-EU shore areas are Russian: the Saint Petersburg area and the exclave of the Kaliningrad Oblast.",
            "is_supporting": true
        },
        {
            "idx": 17,
            "title": "Estonia",
            "paragraph_text": "The Oeselians or Osilians (Estonian saarlased; singular: saarlane) were a historical subdivision of Estonians inhabiting Saaremaa (Danish: Øsel; German: Ösel; Swedish: Ösel), an Estonian island in the Baltic Sea.",
            "is_supporting": true
        }
    ],
    "question": "Which major Russian city borders the body of water in which Saaremaa is located?",
    "question_decomposition": [
        {
            "id": 28482,
            "question": "Where is Saaremaa located?",
            "answer": "the Baltic Sea",
            "paragraph_support_idx": 17
        },
        {
            "id": 46077,
            "question": "which major russian city borders #1",
            "answer": "Saint Petersburg",
            "paragraph_support_idx": 4
        }
    ],
    "answer": "Saint Petersburg",
    "answer_aliases": [
        "Petersburg"
    ],
    "answerable": true
}
```

---
### Corpus 版本格式

Corpus 版本将所有文档/段落提取为独立的语料库，用于检索系统：

```json
{
    "id": "chunk_id",
    "title": "Document Title",
    "text": "Full text content of the document or paragraph.",
    "source": "hotpotqa" | "2wikimultihopqa" | "musique"
}
```

**字段说明**：

- `id`：文档/段落的唯一标识符
- `title`：文档标题
- `text`：文档完整文本内容
- `source`：来源数据集

### 答案格式

模型输出的答案格式：

```json
{
    "id": "<问题的ID值>",
    "original_question": "<原始问题的文本内容>",
    "ground_truth": "<原始问题的答案>",
    "final_answer": "<最终答案的文本内容，如果解析或获取失败则为'Unavailable'>",
    "Inference_process": "<推理过程的描述，如果解析或获取失败则为'Unavailable'>",
    "sub_questions": [
        {
            "sub_question": "<第一个子问题的文本内容>",
            "relevant_chunks": [
                {
                    "title": "<相关文本块的标题>",
                    "chunk": [
                        "<相关文本块的内容片段1>",
                        "<相关文本块的内容片段2>"
                    ]
                }
            ],
            "answer": "<第一个子问题的答案文本内容，如果获取失败则为None>"
        }
    ]
}
```

---

## 问题类型说明

### 1. 比较问题（Comparison Question）

**定义**：对同一组中的两个或多个实体在某些方面进行比较的问题。

**示例**：
- "Who was born first, Albert Einstein or Abraham Lincoln?"（阿尔伯特·爱因斯坦和亚伯拉罕·林肯谁先出生？）

**推理过程**：需要理解问题中的属性（如出生日期），并对两个实体进行定量或逻辑比较来得出答案。

### 2. 推理问题（Inference Question）

**定义**：基于知识库中的两个三元组，利用逻辑规则获取新的三元组，然后根据新三元组创建问题。

**示例**：
- 已知三元组：
  - (Abraham Lincoln, mother, Nancy Hanks Lincoln)
  - (Nancy Hanks Lincoln, father, James Hanks)
- 可得到新三元组：(Abraham Lincoln, maternal grandfather, James Hanks)
- 问题："Who is the maternal grandfather of Abraham Lincoln?"（亚伯拉罕·林肯的外祖父是谁？）
- 答案：James Hanks

**推理过程**：要求系统理解多个逻辑规则，例如要找到 "grandchild"（孙辈），需先找到 "child"（子女），再基于此继续寻找下一级 "child"。

### 3. 组合问题（Compositional Question）

**定义**：由知识库中的两个三元组创建，但与推理问题不同，两个关系不存在推理关系。

**示例**：
- 三元组：
  - (La La Land, distributor, Summit Entertainment)
  - (Summit Entertainment, founded by, Bernd Eichinger)
- 问题："Who is the founder of the company that distributed La La Land film?"（发行《爱乐之城》电影的公司的创始人是谁？）
- 答案：Bernd Eichinger

**推理过程**：系统需要回答多个原始问题并将它们组合起来。如回答上述示例问题，需先回答 "Who is the distributor of La La Land?"（《爱乐之城》的发行商是谁？），再回答其创始人是谁。

### 4. 桥接比较问题（Bridge-Comparison Question）

**定义**：将桥接问题与比较问题相结合，需要找到桥接实体并进行比较以获得最终答案。

**示例**：
- "Which movie has the director born first, La La Land or Tenet?"（《爱乐之城》和《信条》哪部电影的导演出生更早？）

**推理过程**：模型需要找到连接两个段落（一个关于电影，一个关于导演）的桥接实体，获取出生日期信息，然后进行比较得出最终答案。

---

## 参考文献

1. Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., & Manning, C. D. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. *arXiv preprint arXiv:1809.09600*.

2. Ho, X. V., Nguyen, A. D. B., Sugawara, S., & Aizawa, A. (2020). Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps. In *Proceedings of the 28th International Conference on Computational Linguistics* (pp. 6609-6625).

3. Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. (2022). ♫ MuSiQue: Multihop Questions via Single-hop Question Composition. *Transactions of the Association for Computational Linguistics*, 10, 539-554.

4. Zhuang, Y., et al. (2024). Efficient RAG: A Survey on Retrieval-Augmented Generation Efficiency.

---

## 数据集下载

- **HotPotQA**: [https://hotpotqa.github.io/](https://hotpotqa.github.io/)
- **2WikiMultiHopQA**: [https://github.com/Alab-NII/2wikimultihop](https://github.com/Alab-NII/2wikimultihop)
- **MuSiQue**: [https://github.com/StonyBrookNLP/musique](https://github.com/StonyBrookNLP/musique)
