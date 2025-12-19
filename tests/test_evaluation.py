# ============================================================================
# File: tests/test_evaluation.py
# ============================================================================
"""Unit tests for evaluation metrics with statistical significance testing."""

import pytest
import numpy as np
from ralfs.evaluation.faithfulness import (
    EntityGrid,
    EntityGridFaithfulness,
    compute_egf,
)
from ralfs.evaluation.metrics import evaluate_rouge, evaluate_bertscore
from ralfs.evaluation.main import RALFSEvaluator, compare_systems, EvaluationResult


class TestEntityGrid:
    """Test entity grid construction."""
    
    def test_entity_grid_construction(self):
        """Test basic entity grid."""
        text = "Apple announced iPhone. Apple said iPhone is innovative."
        
        grid = EntityGrid(text)
        
        assert grid.grid is not None
        assert len(grid.grid) > 0
        assert 'apple' in grid.grid or 'Apple' in [k.lower() for k in grid.grid.keys()]
    
    def test_entity_roles(self):
        """Test grammatical role assignment."""
        text = "The cat ate the mouse. The dog saw the cat."
        
        grid = EntityGrid(text)
        
        # Should have entities for cat, mouse, dog
        entities = {k.lower() for k in grid.grid.keys()}
        assert 'cat' in entities
    
    def test_transitions(self):
        """Test transition computation."""
        text = "Apple announced iPhone. Apple said iPhone is great. iPhone has features."
        
        grid = EntityGrid(text)
        transitions = grid.transitions
        
        assert len(transitions) > 0
        # Should have some coherent transitions (SS, SO, etc.)


class TestEGFMetric:
    """Test EGF faithfulness metric."""
    
    def test_perfect_match(self):
        """Test identical texts."""
        text = "The cat sat on the mat."
        
        egf = compute_egf(text, text)
        assert egf > 0.9  # Should be very high
    
    def test_entity_preservation(self):
        """Test entity preservation scoring."""
        reference = "Apple announced iPhone. Google released Pixel."
        generated = "Apple unveiled iPhone. Google launched Pixel."
        
        egf = compute_egf(reference, generated)
        assert egf > 0.5  # Good entity preservation
    
    def test_entity_hallucination(self):
        """Test hallucinated entities."""
        reference = "Apple announced iPhone."
        generated = "Microsoft released Windows. Google launched Android."
        
        egf = compute_egf(reference, generated)
        assert egf < 0.5  # Low due to hallucinations
    
    def test_detailed_output(self):
        """Test detailed EGF output."""
        reference = "The company announced earnings. The company exceeded expectations."
        generated = "The company reported earnings. The company beat forecasts."
        
        result = compute_egf(reference, generated, return_details=True)
        
        assert 'egf' in result
        assert 'entity_overlap' in result
        assert 'transition_similarity' in result
        assert 'coherence' in result
        assert 'ref_entities' in result
        assert 'gen_entities' in result


class TestROUGE:
    """Test ROUGE metrics."""
    
    def test_rouge_basic(self):
        """Test basic ROUGE computation."""
        preds = ["The cat sat on the mat"]
        refs = ["The cat was on the mat"]
        
        scores = evaluate_rouge(preds, refs)
        
        assert 0 < scores['rouge1'] < 1
        assert 0 < scores['rougeL'] < 1
    
    def test_rouge_multiple_samples(self):
        """Test ROUGE with multiple samples."""
        preds = [
            "The quick brown fox",
            "Machine learning is great",
        ]
        refs = [
            "A quick brown fox",
            "Deep learning is amazing",
        ]
        
        scores = evaluate_rouge(preds, refs)
        
        assert all(0 <= v <= 1 for v in scores.values())


class TestBERTScore:
    """Test BERTScore metrics."""
    
    def test_bertscore_basic(self):
        """Test basic BERTScore computation."""
        preds = ["The cat sat"]
        refs = ["The cat sits"]
        
        scores = evaluate_bertscore(preds, refs)
        
        assert 0 < scores['bertscore_f1'] < 1
    
    def test_bertscore_semantic_similarity(self):
        """Test semantic similarity."""
        preds = ["The dog is happy"]
        refs = ["The canine is joyful"]
        
        scores = evaluate_bertscore(preds, refs)
        
        # Should capture semantic similarity
        assert scores['bertscore_f1'] > 0.5


class TestEvaluatorWithConfidenceIntervals:
    """Test evaluator with bootstrap confidence intervals."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = RALFSEvaluator(metrics=['rouge', 'bertscore', 'egf'])
        assert evaluator is not None
    
    def test_confidence_intervals(self):
        """Test bootstrap confidence interval computation."""
        predictions = [
            {'id': f'test{i}', 'summary': f'Summary {i}'}
            for i in range(20)
        ]
        references = [
            {'id': f'test{i}', 'summary': f'Reference {i}'}
            for i in range(20)
        ]
        
        evaluator = RALFSEvaluator(metrics=['rouge', 'egf'], verbose=False)
        individual, aggregated = evaluator.evaluate(predictions, references)
        
        # Check confidence intervals exist
        assert hasattr(aggregated, 'rouge1_ci_lower')
        assert hasattr(aggregated, 'rouge1_ci_upper')
        assert aggregated.rouge1_ci_lower <= aggregated.rouge1_mean
        assert aggregated.rouge1_mean <= aggregated.rouge1_ci_upper
    
    def test_bootstrap_ci_computation(self):
        """Test bootstrap CI helper function."""
        evaluator = RALFSEvaluator(metrics=['rouge'], verbose=False)
        
        # Test with known distribution
        scores = [0.5] * 100  # Constant scores
        ci = evaluator._bootstrap_ci(scores)
        
        # CI should be tight around 0.5
        assert 0.49 <= ci[0] <= 0.51
        assert 0.49 <= ci[1] <= 0.51
    
    def test_bootstrap_ci_variance(self):
        """Test bootstrap CI with varying scores."""
        evaluator = RALFSEvaluator(metrics=['rouge'], verbose=False)
        
        # Scores with variance
        np.random.seed(42)
        scores = list(np.random.uniform(0.3, 0.7, 100))
        ci = evaluator._bootstrap_ci(scores)
        
        # CI should be wider than constant case
        width = ci[1] - ci[0]
        assert width > 0.01


class TestStatisticalSignificance:
    """Test statistical significance testing."""
    
    def test_compare_systems_basic(self):
        """Test basic system comparison."""
        # System 1 (baseline)
        sys1_results = [
            EvaluationResult(
                id=f'test{i}',
                reference=f'ref{i}',
                prediction=f'pred1_{i}',
                rouge1=0.5,
                rouge2=0.3,
                rougeL=0.4,
                bertscore_f1=0.6,
                egf=0.5,
            )
            for i in range(30)
        ]
        
        # System 2 (improved)
        sys2_results = [
            EvaluationResult(
                id=f'test{i}',
                reference=f'ref{i}',
                prediction=f'pred2_{i}',
                rouge1=0.6,  # +0.1 improvement
                rouge2=0.4,  # +0.1 improvement
                rougeL=0.5,  # +0.1 improvement
                bertscore_f1=0.7,  # +0.1 improvement
                egf=0.6,  # +0.1 improvement
            )
            for i in range(30)
        ]
        
        comparison = compare_systems(sys1_results, sys2_results)
        
        # Check that differences are computed
        assert 'rouge1_diff_mean' in comparison
        assert 'rouge1_p_value' in comparison
        assert 'rouge1_significant' in comparison
        
        # Should detect improvement
        assert comparison['rouge1_diff_mean'] > 0
        assert comparison['rougeL_diff_mean'] > 0
    
    def test_compare_systems_no_difference(self):
        """Test comparison when systems are equivalent."""
        # Both systems produce same results
        sys1_results = [
            EvaluationResult(
                id=f'test{i}',
                reference=f'ref{i}',
                prediction=f'pred_{i}',
                rouge1=0.5,
                rouge2=0.3,
                rougeL=0.4,
                bertscore_f1=0.6,
                egf=0.5,
            )
            for i in range(30)
        ]
        
        sys2_results = sys1_results.copy()
        
        comparison = compare_systems(sys1_results, sys2_results)
        
        # Differences should be near zero
        assert abs(comparison['rouge1_diff_mean']) < 0.001
        assert abs(comparison['rougeL_diff_mean']) < 0.001
        
        # Should not be significant
        assert not comparison['rouge1_significant']
    
    def test_cohens_d_calculation(self):
        """Test effect size (Cohen's d) computation."""
        sys1_results = [
            EvaluationResult(
                id=f'test{i}',
                reference=f'ref{i}',
                prediction=f'pred1_{i}',
                rouge1=0.5,
                rouge2=0.3,
                rougeL=0.4,
                bertscore_f1=0.6,
                egf=0.5,
            )
            for i in range(50)
        ]
        
        sys2_results = [
            EvaluationResult(
                id=f'test{i}',
                reference=f'ref{i}',
                prediction=f'pred2_{i}',
                rouge1=0.7,  # Large improvement
                rouge2=0.5,
                rougeL=0.6,
                bertscore_f1=0.8,
                egf=0.7,
            )
            for i in range(50)
        ]
        
        comparison = compare_systems(sys1_results, sys2_results)
        
        # Cohen's d should be computed
        assert 'rouge1_cohens_d' in comparison
        # Large effect size for 0.2 difference
        assert abs(comparison['rouge1_cohens_d']) > 0


class TestEvaluator:
    """Test complete evaluator."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = RALFSEvaluator(metrics=['rouge', 'bertscore', 'egf'])
        assert evaluator is not None
    
    def test_evaluate_single_sample(self):
        """Test evaluation on single sample."""
        predictions = [
            {'id': 'test1', 'summary': 'The cat sat on the mat.'}
        ]
        references = [
            {'id': 'test1', 'summary': 'The cat was on the mat.'}
        ]
        
        evaluator = RALFSEvaluator(metrics=['rouge', 'egf'], verbose=False)
        individual, aggregated = evaluator.evaluate(predictions, references)
        
        assert len(individual) == 1
        assert aggregated.num_samples == 1
        assert 0 <= aggregated.rouge1_mean <= 1
    
    def test_evaluate_multiple_samples(self):
        """Test evaluation on multiple samples."""
        predictions = [
            {'id': f'test{i}', 'summary': f'Summary {i}'}
            for i in range(5)
        ]
        references = [
            {'id': f'test{i}', 'summary': f'Reference {i}'}
            for i in range(5)
        ]
        
        evaluator = RALFSEvaluator(metrics=['rouge'], verbose=False)
        individual, aggregated = evaluator.evaluate(predictions, references)
        
        assert len(individual) == 5
        assert aggregated.num_samples == 5


@pytest.mark.slow
class TestEvaluationIntegration:
    """Integration tests for evaluation pipeline."""
    
    def test_full_evaluation_pipeline(self, tmp_path):
        """Test complete evaluation pipeline."""
        from ralfs.evaluation.main import run_evaluation
        from ralfs.utils.io import save_json
        
        # Create test data
        predictions = [
            {'id': 'test1', 'summary': 'Test summary 1'},
            {'id': 'test2', 'summary': 'Test summary 2'},
        ]
        references = [
            {'id': 'test1', 'summary': 'Reference summary 1'},
            {'id': 'test2', 'summary': 'Reference summary 2'},
        ]
        
        # Save to files
        pred_path = tmp_path / "predictions.json"
        ref_path = tmp_path / "references.json"
        save_json(predictions, pred_path)
        save_json(references, ref_path)
        
        # Run evaluation
        output_dir = tmp_path / "evaluation"
        results = run_evaluation(
            pred_path,
            ref_path,
            output_dir,
            metrics=['rouge', 'egf'],
        )
        
        assert results.num_samples == 2
        assert (output_dir / "eval_aggregated.json").exists()
        assert (output_dir / "eval_individual.json").exists()
        assert (output_dir / "eval_latex.tex").exists()
