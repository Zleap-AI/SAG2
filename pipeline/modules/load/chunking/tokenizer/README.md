# Chunking Tokenizer Assets

把真实 tokenizer 文件放到：

- `pipeline/modules/load/chunking/tokenizer/assets/tokenizer.json`

也可以通过环境变量指定路径：

- `DATAFLOW_CHUNKING_TOKENIZER_JSON`
- `DATAFLOW_TOKENIZER_JSON`

`MarkdownTextChunker` 和 `MarkdownSourceChunkAssembler` 会直接使用该 tokenizer 做 token 精确计数。
