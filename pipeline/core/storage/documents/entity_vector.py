"""
实体向量 Document 模型

对应 Elasticsearch 索引：entity_vectors
"""

from elasticsearch_dsl import Boolean, Date, DenseVector, Document, Keyword, Text


class EntityVectorDocument(Document):
    """实体向量文档模型"""

    # 字段定义
    entity_id = Keyword(required=True)
    source_config_id = Keyword(required=True)
    type = Keyword(required=True)  # 实体类型：PERSON, ORGANIZATION, TOPIC等
    name = Text(fields={"keyword": Keyword()})
    vector = DenseVector(dims=1024, index=True, similarity="cosine")
    created_time = Date()
    is_delete = Boolean()  # 软删除标记

    class Index:
        """索引配置"""

        name = "entity_vectors"
        settings = {"number_of_shards": 24, "number_of_replicas": 1}

    def save(self, **kwargs):
        """保存文档"""
        return super().save(**kwargs)
