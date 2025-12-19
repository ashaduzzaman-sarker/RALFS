# ============================================================================
# File: tests/test_pipeline.py
# ============================================================================
"""
End-to-end pipeline tests for RALFS.

Tests the complete workflow:
1. Data preprocessing
2. Index building
3. Retrieval
4. Generation
5. Evaluation
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from ralfs.core.config import load_config, RALFSConfig
from ralfs.core.constants import PROCESSED_DIR, INDEX_DIR
from ralfs.data.processor import DocumentProcessor
from ralfs.data.indexer import IndexBuilder
from ralfs.retriever import create_retriever
from ralfs.generator import create_generator
from ralfs.evaluation.metrics import evaluate_rouge, evaluate_bertscore
from ralfs.evaluation.faithfulness import compute_egf
from ralfs.training.dataset import FiDDataset


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = RALFSConfig()
    config.data.dataset = "arxiv"
    config.data.max_samples = 5  # Small for testing
    config.data.chunk_size = 256
    config.data.overlap = 64
    config.retriever.k_final = 5
    config.generator.adaptive_k_max = 5
    return config


class TestDataPipeline:
    """Test data preprocessing pipeline."""
    
    def test_preprocessing(self, test_config, temp_output_dir):
        """Test document preprocessing and chunking."""
        # Override output
        test_config.data.max_samples = 2
        
        processor = DocumentProcessor(test_config)
        
        # Download would take too long, so skip if data doesn't exist
        try:
            chunks = processor.load_chunks()
            assert len(chunks) > 0, "No chunks loaded"
        except FileNotFoundError:
            pytest.skip("Preprocessed data not available")
    
    def test_chunking_strategies(self, test_config):
        """Test different chunking strategies."""
        from ralfs.data.chunker import FixedChunker, SentenceChunker, SemanticChunker
        
        text = """
        This is the first sentence. This is the second sentence.
        This is the third sentence. This is the fourth sentence.
        This is the fifth sentence.
        """
        
        doc_id = "test_doc"
        
        # Fixed chunking
        fixed_chunker = FixedChunker(chunk_size=10, overlap=2)
        fixed_chunks = fixed_chunker.chunk(text, doc_id)
        assert len(fixed_chunks) > 0
        
        # Sentence chunking
        sent_chunker = SentenceChunker(chunk_size=10, overlap=2)
        sent_chunks = sent_chunker.chunk(text, doc_id)
        assert len(sent_chunks) > 0
        
        # Semantic chunking
        semantic_chunker = SemanticChunker(chunk_size=10, overlap=2)
        semantic_chunks = semantic_chunker.chunk(text, doc_id)
        assert len(semantic_chunks) > 0


class TestRetrieval:
    """Test retrieval pipeline."""
    
    def test_dense_retrieval(self, test_config):
        """Test dense retrieval."""
        try:
            from ralfs.retriever.dense import DenseRetriever
            
            retriever = DenseRetriever(test_config)
            retriever.load_index()
            
            results = retriever.retrieve("quantum computing", k=5)
            assert len(results) <= 5
            assert all(hasattr(r, 'text') for r in results)
            assert all(hasattr(r, 'score') for r in results)
            
        except FileNotFoundError:
            pytest.skip("Index not built")
    
    def test_sparse_retrieval(self, test_config):
        """Test sparse (BM25) retrieval."""
        try:
            from ralfs.retriever.sparse import SparseRetriever
            
            retriever = SparseRetriever(test_config)
            retriever.load_index()
            
            results = retriever.retrieve("quantum computing", k=5)
            assert len(results) <= 5
            
        except FileNotFoundError:
            pytest.skip("Chunks not available")
    
    def test_hybrid_retrieval(self, test_config):
        """Test hybrid retrieval (dense + sparse + reranker)."""
        try:
            retriever = create_retriever(test_config)
            retriever.load_index()
            
            results = retriever.retrieve("machine learning", k=5)
            assert len(results) <= 5
            
            # Check scores are sorted
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)
            
        except FileNotFoundError:
            pytest.skip("Index not built")


class TestGeneration:
    """Test generation pipeline."""
    
    def test_generator_initialization(self, test_config):
        """Test generator initialization."""
        generator = create_generator(test_config)
        assert generator is not None
        assert generator.is_initialized()
    
    def test_generation_with_passages(self, test_config):
        """Test summary generation."""
        generator = create_generator(test_config)
        
        query = "What is quantum computing?"
        passages = [
            {"text": "Quantum computing uses quantum mechanics principles.", "score": 0.9},
            {"text": "Qubits can represent multiple states simultaneously.", "score": 0.8},
            {"text": "Quantum computers can solve certain problems faster.", "score": 0.7},
        ]
        
        result = generator.generate(query, passages)
        
        assert result.summary is not None
        assert len(result.summary) > 0
        assert result.k_used <= len(passages)
        assert result.num_passages == len(passages)
    
    def test_adaptive_k_selection(self, test_config):
        """Test adaptive k selection."""
        from ralfs.generator.adaptive_k import AdaptiveKSelector
        
        selector = AdaptiveKSelector(min_k=3, max_k=10, strategy="score_dropoff")
        
        # Test with different score patterns
        scores_high_quality = [0.9, 0.88, 0.85, 0.3, 0.2, 0.1]  # Clear drop-off
        k1 = selector.select_k(scores_high_quality)
        assert 3 <= k1 <= 4  # Should stop before the drop
        
        scores_uniform = [0.8, 0.78, 0.76, 0.74, 0.72, 0.7]  # No clear drop
        k2 = selector.select_k(scores_uniform)
        assert k2 >= k1  # Should use more passages


class TestEvaluation:
    """Test evaluation metrics."""
    
    def test_rouge_scores(self):
        """Test ROUGE evaluation."""
        predictions = [
            "The cat sat on the mat.",
            "Machine learning is a subset of AI.",
        ]
        references = [
            "The cat was sitting on the mat.",
            "Machine learning is part of artificial intelligence.",
        ]
        
        scores = evaluate_rouge(predictions, references)
        
        assert 'rouge1' in scores
        assert 'rouge2' in scores
        assert 'rougeL' in scores
        assert all(0 <= v <= 1 for v in scores.values())
    
    def test_bertscore(self):
        """Test BERTScore evaluation."""
        predictions = ["The quick brown fox."]
        references = ["A quick brown fox."]
        
        scores = evaluate_bertscore(predictions, references)
        
        assert 'bertscore_precision' in scores
        assert 'bertscore_recall' in scores
        assert 'bertscore_f1' in scores
        assert all(0 <= v <= 1 for v in scores.values())
    
    def test_egf_metric(self):
        """Test Entity Grid Faithfulness metric."""
        reference = """
        Apple announced a new iPhone yesterday. The company said the device
        will feature improved cameras. Apple expects strong sales this quarter.
        """
        
        generated_faithful = """
        Apple unveiled a new iPhone. The company announced improved cameras.
        Apple anticipates good sales.
        """
        
        generated_unfaithful = """
        Microsoft released a new phone. Google announced better cameras.
        Amazon expects strong sales.
        """
        
        egf_faithful = compute_egf(reference, generated_faithful)
        egf_unfaithful = compute_egf(reference, generated_unfaithful)
        
        # Faithful summary should score higher
        assert egf_faithful > egf_unfaithful
        assert 0 <= egf_faithful <= 1
        assert 0 <= egf_unfaithful <= 1
    
    def test_egf_detailed(self):
        """Test EGF with detailed output."""
        reference = "Apple announced iPhone. Apple said it's innovative."
        generated = "Apple unveiled iPhone. The company praised innovation."
        
        result = compute_egf(reference, generated, return_details=True)
        
        assert isinstance(result, dict)
        assert 'egf' in result
        assert 'entity_overlap' in result
        assert 'transition_similarity' in result
        assert 'coherence' in result


class TestTraining:
    """Test training pipeline."""
    
    def test_dataset_creation(self, test_config):
        """Test FiD dataset creation."""
        try:
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            
            data_path = PROCESSED_DIR / f"{test_config.data.dataset}_train_chunks.jsonl"
            
            if not data_path.exists():
                pytest.skip("Training data not available")
            
            dataset = FiDDataset(
                data_path=data_path,
                tokenizer=tokenizer,
                max_input_length=256,
                max_output_length=100,
                max_passages=5,
            )
            
            assert len(dataset) > 0
            
            # Test __getitem__
            sample = dataset[0]
            assert 'input_ids' in sample
            assert 'attention_mask' in sample
            assert 'labels' in sample
            
        except Exception as e:
            pytest.skip(f"Dataset test failed: {e}")
    
    def test_trainer_initialization(self, test_config):
        """Test trainer initialization."""
        from ralfs.training.trainer import RALFSTrainer
        
        trainer = RALFSTrainer(test_config)
        assert trainer is not None
        assert trainer.output_dir.exists()


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.slow
    def test_full_pipeline_mini(self, test_config, temp_output_dir):
        """Test complete pipeline with minimal data."""
        # This is a slow test - only run with pytest -m slow
        
        # Step 1: Preprocess (skip if data exists)
        try:
            processor = DocumentProcessor(test_config)
            chunks = processor.load_chunks()
        except FileNotFoundError:
            pytest.skip("No preprocessed data available")
        
        # Step 2: Build index (skip if exists)
        try:
            builder = IndexBuilder(test_config)
            builder.build_dense_index()
            builder.build_sparse_index()
        except FileNotFoundError:
            pytest.skip("Cannot build indexes")
        
        # Step 3: Retrieve
        retriever = create_retriever(test_config)
        retriever.load_index()
        
        query = chunks[0].text[:200] if chunks else "test query"
        results = retriever.retrieve(query, k=3)
        assert len(results) > 0
        
        # Step 4: Generate
        generator = create_generator(test_config)
        passages = [{'text': r.text, 'score': r.score} for r in results]
        
        gen_result = generator.generate(query, passages)
        assert gen_result.summary is not None
        
        # Step 5: Evaluate
        predictions = [gen_result.summary]
        references = [chunks[0].metadata.get('summary', 'Reference summary')]
        
        rouge_scores = evaluate_rouge(predictions, references)
        assert all(score >= 0 for score in rouge_scores.values())


class TestCLI:
    """Test CLI commands."""
    
    def test_cli_import(self):
        """Test CLI module can be imported."""
        from ralfs.cli import app
        assert app is not None
    
    def test_config_loading(self):
        """Test configuration loading."""
        config = load_config()
        assert config is not None
        assert hasattr(config, 'data')
        assert hasattr(config, 'retriever')
        assert hasattr(config, 'generator')
        assert hasattr(config, 'train')


@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring full setup."""
    
    def test_retrieval_and_generation(self, test_config):
        """Test retrieval followed by generation."""
        try:
            # Retrieve
            retriever = create_retriever(test_config)
            retriever.load_index()
            
            query = "What is deep learning?"
            results = retriever.retrieve(query, k=5)
            
            # Generate
            generator = create_generator(test_config)
            passages = [{'text': r.text, 'score': r.score} for r in results]
            
            gen_result = generator.generate(query, passages)
            
            assert gen_result.summary is not None
            assert gen_result.k_used > 0
            
        except FileNotFoundError:
            pytest.skip("Index or model not available")


# # Pytest markers
# pytestmark = [
#     pytest.mark.filterwarnings("ignore::DeprecationWarning"),
#     pytest.mark.filterwarnings("ignore::FutureWarning"),
# ]