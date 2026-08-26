"""Tests for NVIDIA NIM LLM provider integration."""

import os
import pytest
from src.utils.llm_factory import get_llm


def test_nvidia_llm_instantiation():
    llm = get_llm(
        provider="nvidia",
        model_name="nvidia/nemotron-3-super-120b-a12b",
        api_key="nvapi-test-key"
    )
    assert llm is not None
    assert getattr(llm, "model_name", None) == "nvidia/nemotron-3-super-120b-a12b"
