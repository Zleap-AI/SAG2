import re
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict, List, Tuple, Union

from pipeline.utils import get_logger
from pipeline.core.ai.factory import create_llm_client, get_embedding_client
from ..utils.eval_utils import normalize_answer

logger = get_logger("evaluation.metrics.base")


class BaseMetric:
    metric_name = "base"


    def __init__(self,) -> None:

        logger.debug(f"Loading {self.__class__.__name__}")
        self._llm_client = None
        self._embedding_client = None


    async def get_llm_client(self, scenario: str = "general", model_config: Optional[Dict] = None):
        """
        Get LLM client using project's factory method.

        Args:
            scenario: LLM scenario (general, extract, search, chat, summary)
            model_config: Optional model configuration override

        Returns:
            LLM client instance
        """
        if self._llm_client is None or model_config is not None:
            self._llm_client = await create_llm_client(
                scenario=scenario,
                model_config=model_config
            )
        return self._llm_client

    async def get_embedding_client(self, scenario: str = "general"):
        """
        Get Embedding client using project's factory method.

        Args:
            scenario: Embedding scenario

        Returns:
            Embedding client instance
        """
        if self._embedding_client is None:
            self._embedding_client = await get_embedding_client(scenario=scenario)
        return self._embedding_client

    def calculate_metric_scores(self) -> Tuple[Dict[str, Union[int, float]], List[Union[int, float]]]:
        """
        Calculate the total score under this metric and score for each individual example in the input.


        Returns:
            Tuple[Dict[str, Union[int, float]], List[Union[int, float]]]
        """
        return {}, []
    

    