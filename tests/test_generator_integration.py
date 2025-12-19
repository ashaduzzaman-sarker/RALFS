# ============================================================================
# File: tests/test_generator_integration.py
# ============================================================================
"""Integration tests for generator with retrieval."""

import pytest
from ralfs.core.config import RALFSConfig, DataConfig, RetrieverConfig, GeneratorConfig
from ralfs.generator import create_generator


@pytest.fixture
def integration_config():
    """Create configuration for integration test."""
    config = RALFSConfig()
    config.data = DataConfig(dataset="arxiv", split="train", max_samples=5)
    config.retriever = RetrieverConfig(type="dense", k_dense=10)
    config.generator = GeneratorConfig(
        model_name="google/flan-t5-small",
        max_input_length=256,
        max_output_length=50,
    )
    return config


@pytest.mark.slow
@pytest.mark.integration
class TestGeneratorIntegration:
    """Integration tests for full retrieval + generation pipeline."""
    
    def test_retrieval_to_generation_pipeline(self, integration_config):
        """Test complete pipeline: preprocess → retrieve → generate."""
        # Step 1: Preprocess
        from ralfs.data.processor import run_preprocessing
        from ralfs.data.indexer import build_index
        
        run_preprocessing(integration_config, force_rechunk=True)
        build_index(integration_config, force_rebuild=True)
        
        # Step 2: Retrieve
        from ralfs.retriever import create_retriever
        
        retriever = create_retriever(integration_config, retriever_type="dense")
        retriever.load_index()
        results = retriever.retrieve("machine learning", k=10)
        
        # Convert to passages format
        passages = [
            {"text": r.text, "score": r.score}
            for r in results
        ]
        
        # Step 3: Generate
        generator = create_generator(integration_config)
        gen_result = generator.generate("machine learning", passages)
        
        assert isinstance(gen_result.summary, str)
        assert len(gen_result.summary) > 0
        assert gen_result.k_used > 0
        assert gen_result.k_used <= len(passages)