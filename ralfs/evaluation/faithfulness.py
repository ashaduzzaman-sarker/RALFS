# ============================================================================
# File: ralfs/evaluation/faithfulness.py
# ============================================================================
"""
Entity Grid Faithfulness (EGF) - Novel coherence and faithfulness metric.

This metric evaluates summary faithfulness by analyzing entity transitions
across sentences, inspired by the Barzilay & Lapata (2008) entity grid model.

Mathematical Formulation:
    EGF = α·E_overlap + β·T_similarity + γ·C_score
    
    where:
    - E_overlap: Jaccard similarity of entities between reference and generated
    - T_similarity: 1 - JS_divergence(P_ref || P_gen) for transition distributions
    - C_score: Normalized coherence score based on transition quality
    - α=0.4, β=0.4, γ=0.2 (empirically tuned weights)

Entity transitions are classified as:
    - Coherent: {SS, SO, SX, OS, OO} (maintain entity focus)
    - Incoherent: {--, -S, -O, -X, S-, O-, X-} (abrupt changes)

Citation:
    Barzilay, R., & Lapata, M. (2008). Modeling local coherence: An entity-based approach.
    Computational Linguistics, 34(1), 1-34.
"""

from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import spacy
from ralfs.core.logging import get_logger

logger = get_logger(__name__)

# Load spacy model (singleton)
_nlp = None

def get_nlp():
    """Get or load spaCy model (singleton pattern)."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            logger.error(
                "spaCy model 'en_core_web_sm' not found. "
                "Install with: python -m spacy download en_core_web_sm"
            )
            raise
    return _nlp


@dataclass
class EntityMention:
    """Entity mention with grammatical role."""
    entity: str  # Entity text
    role: str    # Grammatical role: S (subject), O (object), X (other)
    sentence_idx: int


class EntityGrid:
    """
    Entity Grid for coherence analysis.
    
    Represents entities and their grammatical roles across sentences.
    Grid format: entities (rows) × sentences (columns)
    """
    
    def __init__(self, text: str):
        """
        Build entity grid from text.
        
        Args:
            text: Input text (reference or generated summary)
        """
        self.text = text
        self.nlp = get_nlp()
        
        # Parse text
        self.doc = self.nlp(text)
        self.sentences = list(self.doc.sents)
        
        # Build grid
        self.grid = self._build_grid()
        
        # Compute transitions
        self.transitions = self._compute_transitions()
    
    def _build_grid(self) -> Dict[str, List[str]]:
        """
        Build entity grid: entity -> [role_sent1, role_sent2, ...].
        
        Returns:
            Dict mapping entity to list of roles per sentence
        """
        grid = defaultdict(lambda: ["-"] * len(self.sentences))
        
        for sent_idx, sent in enumerate(self.sentences):
            # Extract entities with roles
            mentions = self._extract_mentions(sent, sent_idx)
            
            for mention in mentions:
                # Use lemmatized form for entity matching
                entity_key = mention.entity.lower()
                
                # Assign role (priority: S > O > X)
                current_role = grid[entity_key][sent_idx]
                if current_role == "-" or self._role_priority(mention.role) > self._role_priority(current_role):
                    grid[entity_key][sent_idx] = mention.role
        
        return dict(grid)
    
    def _extract_mentions(self, sent: spacy.tokens.Span, sent_idx: int) -> List[EntityMention]:
        """
        Extract entity mentions with grammatical roles from sentence.
        
        Args:
            sent: spaCy sentence span
            sent_idx: Sentence index
        
        Returns:
            List of EntityMention objects
        """
        mentions = []
        
        # Extract named entities
        for ent in sent.ents:
            role = self._get_grammatical_role(ent)
            mentions.append(EntityMention(
                entity=ent.lemma_,
                role=role,
                sentence_idx=sent_idx,
            ))
        
        # Extract noun chunks (for non-named entities)
        for chunk in sent.noun_chunks:
            # Skip if already covered by named entity
            if any(chunk.start <= ent.start < chunk.end for ent in sent.ents):
                continue
            
            role = self._get_grammatical_role(chunk.root)
            mentions.append(EntityMention(
                entity=chunk.root.lemma_,
                role=role,
                sentence_idx=sent_idx,
            ))
        
        return mentions
    
    def _get_grammatical_role(self, token_or_span) -> str:
        """
        Determine grammatical role: S (subject), O (object), X (other).
        
        Args:
            token_or_span: spaCy token or span
        
        Returns:
            Role string: "S", "O", or "X"
        """
        # Get root token
        if hasattr(token_or_span, 'root'):
            token = token_or_span.root
        else:
            token = token_or_span
        
        # Check dependency
        dep = token.dep_
        
        if dep in ["nsubj", "nsubjpass"]:
            return "S"
        elif dep in ["dobj", "pobj", "iobj"]:
            return "O"
        else:
            return "X"
    
    def _role_priority(self, role: str) -> int:
        """Role priority for conflict resolution."""
        priority = {"S": 3, "O": 2, "X": 1, "-": 0}
        return priority.get(role, 0)
    
    def _compute_transitions(self) -> Dict[str, int]:
        """
        Compute entity role transitions across sentences.
        
        Transition types:
        - SS, SO, SX, S-: Subject transitions
        - OS, OO, OX, O-: Object transitions
        - XS, XO, XX, X-: Other transitions
        - -S, -O, -X: New entity introductions
        
        Returns:
            Dict mapping transition type to count
        """
        transitions = defaultdict(int)
        
        for entity, roles in self.grid.items():
            # Count transitions between consecutive sentences
            for i in range(len(roles) - 1):
                transition = f"{roles[i]}{roles[i+1]}"
                transitions[transition] += 1
        
        return dict(transitions)
    
    def get_transition_probabilities(self) -> Dict[str, float]:
        """
        Compute transition probability distribution.
        
        Returns:
            Dict mapping transition type to probability
        """
        total = sum(self.transitions.values())
        if total == 0:
            return {}
        
        return {
            trans: count / total
            for trans, count in self.transitions.items()
        }


class EntityGridFaithfulness:
    """
    Entity Grid Faithfulness (EGF) metric.
    
    Measures how well a generated summary maintains the entity coherence
    patterns of the reference text.
    
    Higher scores indicate better faithfulness and coherence.
    """
    
    # Coherence-promoting transitions (based on Centering Theory)
    COHERENT_TRANSITIONS = {
        "SS", "SO", "SX",  # Subject continuity (strong coherence)
        "OS", "OO",        # Object continuity (moderate coherence)
    }
    
    # Incoherent transitions
    INCOHERENT_TRANSITIONS = {
        "--", "-S", "-O", "-X",  # Abrupt introductions
        "S-", "O-", "X-",        # Abrupt disappearances
    }
    
    def __init__(self):
        """Initialize EGF metric."""
        self.nlp = get_nlp()
    
    def compute(
        self,
        reference: str,
        generated: str,
        return_details: bool = False,
    ) -> float | Dict[str, any]:
        """
        Compute EGF score between reference and generated text.
        
        Args:
            reference: Reference summary
            generated: Generated summary
            return_details: If True, return detailed breakdown
        
        Returns:
            EGF score (0-1, higher is better) or dict with details
        """
        # Build entity grids
        ref_grid = EntityGrid(reference)
        gen_grid = EntityGrid(generated)
        
        # Compute sub-scores
        entity_overlap = self._entity_overlap_score(ref_grid, gen_grid)
        transition_similarity = self._transition_similarity(ref_grid, gen_grid)
        coherence_score = self._coherence_score(gen_grid)
        
        # Combined EGF score (weighted average)
        egf = (
            0.4 * entity_overlap +       # Entity coverage
            0.4 * transition_similarity + # Transition pattern matching
            0.2 * coherence_score         # Generated text coherence
        )
        
        if return_details:
            return {
                "egf": egf,
                "entity_overlap": entity_overlap,
                "transition_similarity": transition_similarity,
                "coherence": coherence_score,
                "ref_entities": len(ref_grid.grid),
                "gen_entities": len(gen_grid.grid),
                "ref_transitions": ref_grid.transitions,
                "gen_transitions": gen_grid.transitions,
            }
        
        return egf
    
    def _entity_overlap_score(self, ref_grid: EntityGrid, gen_grid: EntityGrid) -> float:
        """
        Compute entity overlap between reference and generated.
        
        Measures how many important entities are preserved.
        """
        ref_entities = set(ref_grid.grid.keys())
        gen_entities = set(gen_grid.grid.keys())
        
        if not ref_entities:
            return 0.0
        
        # Jaccard similarity
        overlap = len(ref_entities & gen_entities)
        union = len(ref_entities | gen_entities)
        
        if union == 0:
            return 0.0
        
        return overlap / union
    
    def _transition_similarity(self, ref_grid: EntityGrid, gen_grid: EntityGrid) -> float:
        """
        Compute similarity between transition distributions.
        
        Uses Jensen-Shannon divergence between transition probabilities.
        """
        ref_probs = ref_grid.get_transition_probabilities()
        gen_probs = gen_grid.get_transition_probabilities()
        
        if not ref_probs or not gen_probs:
            return 0.0
        
        # Get all transition types
        all_transitions = set(ref_probs.keys()) | set(gen_probs.keys())
        
        # Build probability vectors
        ref_vec = np.array([ref_probs.get(t, 0.0) for t in all_transitions])
        gen_vec = np.array([gen_probs.get(t, 0.0) for t in all_transitions])
        
        # Compute Jensen-Shannon divergence
        js_div = self._jensen_shannon_divergence(ref_vec, gen_vec)
        
        # Convert to similarity (1 - divergence)
        similarity = 1.0 - js_div
        
        return max(0.0, similarity)
    
    def _coherence_score(self, grid: EntityGrid) -> float:
        """
        Compute coherence score based on transition quality.
        
        Rewards coherent transitions, penalizes incoherent ones.
        """
        total_transitions = sum(grid.transitions.values())
        
        if total_transitions == 0:
            return 0.0
        
        # Count coherent and incoherent transitions
        coherent = sum(
            grid.transitions.get(t, 0)
            for t in self.COHERENT_TRANSITIONS
        )
        incoherent = sum(
            grid.transitions.get(t, 0)
            for t in self.INCOHERENT_TRANSITIONS
        )
        
        # Coherence ratio
        coherence = (coherent - 0.5 * incoherent) / total_transitions
        
        # Normalize to [0, 1]
        return max(0.0, min(1.0, coherence + 0.5))
    
    def _jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute Jensen-Shannon divergence between two distributions.
        
        JSD is a symmetric and bounded version of KL divergence.
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        
        # Average distribution
        m = 0.5 * (p + q)
        
        # KL divergences
        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))
        
        # JS divergence
        js = 0.5 * (kl_pm + kl_qm)
        
        # Normalize to [0, 1]
        return np.sqrt(js / np.log(2))


# Convenience function
def compute_egf(
    reference: str,
    generated: str,
    return_details: bool = False,
) -> float | Dict[str, any]:
    """
    Compute Entity Grid Faithfulness score.
    
    Args:
        reference: Reference summary
        generated: Generated summary
        return_details: If True, return detailed breakdown
    
    Returns:
        EGF score (0-1, higher is better) or dict with details
    
    Example:
        >>> reference = "Apple announced new iPhone. The company said..."
        >>> generated = "Apple unveiled iPhone. Apple stated..."
        >>> score = compute_egf(reference, generated)
        >>> print(f"EGF: {score:.3f}")
    """
    metric = EntityGridFaithfulness()
    return metric.compute(reference, generated, return_details=return_details)
