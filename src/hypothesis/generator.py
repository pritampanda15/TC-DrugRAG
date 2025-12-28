"""
Hypothesis Generator

Generates drug discovery hypotheses using LLM reasoning
over causally-retrieved evidence.
"""

import json
import logging
import time
from datetime import datetime
from typing import Optional

from ..knowledge_graph.schemas import CausalPath, DrugDiscoveryQuery
from ..retrieval import CausalRetrievalEngine, RetrievalResult
from .structured_output import DrugDiscoveryHypothesis, HypothesisBatch
from .uncertainty import UncertaintyQuantifier

logger = logging.getLogger(__name__)


class HypothesisGenerator:
    """
    Generate drug discovery hypotheses from causal evidence.

    This is the core hypothesis generation component that:
    1. Takes causal evidence from the retrieval engine
    2. Uses an LLM to reason over the evidence
    3. Generates structured hypotheses with mechanisms
    4. Calculates uncertainty bounds for each hypothesis

    Example:
        >>> generator = HypothesisGenerator(retrieval_engine)
        >>> hypotheses = generator.generate(
        ...     drug="Metformin",
        ...     disease="Glioblastoma",
        ...     num_hypotheses=5,
        ... )
        >>> for h in hypotheses.hypotheses:
        ...     print(h.get_summary())
    """

    SYSTEM_PROMPT = """You are an expert drug discovery scientist specializing in
identifying novel therapeutic hypotheses. Your task is to analyze causal evidence
and generate scientifically rigorous hypotheses for drug-disease relationships.

For each hypothesis, you must provide:
1. A clear mechanism of action explaining HOW the drug might treat the disease
2. The key molecular targets involved
3. Supporting evidence from the provided causal paths
4. Potential limitations or uncertainties
5. Suggested validation experiments

Be scientifically rigorous and avoid speculation beyond what the evidence supports.
Clearly distinguish between well-supported claims and more speculative connections."""

    GENERATION_PROMPT = """Based on the following causal evidence, generate {num_hypotheses}
drug discovery hypotheses for using {drug} to treat {disease}.

## Causal Evidence (from knowledge graph):
{causal_paths}

## Supporting Literature:
{documents}

## Evidence Statistics:
- Number of causal paths: {num_paths}
- Number of supporting documents: {num_docs}
- Key entities: {key_entities}
- Consensus mechanisms: {consensus}

Generate hypotheses in the following JSON format:
{{
    "hypotheses": [
        {{
            "drug": "{drug}",
            "target": "primary molecular target",
            "disease": "{disease}",
            "mechanism": "detailed mechanism of action",
            "evidence_summary": ["key evidence point 1", "key evidence point 2"],
            "confidence": "high/medium/low",
            "novelty": "high/medium/low",
            "limitations": ["limitation 1"],
            "suggested_experiments": ["experiment 1", "experiment 2"]
        }}
    ]
}}

Important: Only generate hypotheses that are supported by the provided evidence.
Rank by scientific plausibility and evidence strength."""

    def __init__(
        self,
        retrieval_engine: CausalRetrievalEngine,
        llm_provider: str = "openai",
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
    ):
        """
        Initialize the hypothesis generator.

        Args:
            retrieval_engine: The causal retrieval engine
            llm_provider: LLM provider (openai, anthropic)
            model: Model name/ID
            temperature: Generation temperature
        """
        self.retrieval_engine = retrieval_engine
        self.llm_provider = llm_provider
        self.model = model
        self.temperature = temperature

        self.uncertainty_quantifier = UncertaintyQuantifier()
        self._client = None

    def _get_client(self):
        """Lazy load LLM client."""
        if self._client is None:
            if self.llm_provider == "openai":
                from openai import OpenAI
                self._client = OpenAI()
            elif self.llm_provider == "anthropic":
                from anthropic import Anthropic
                self._client = Anthropic()
            else:
                raise ValueError(f"Unknown provider: {self.llm_provider}")
        return self._client

    def generate(
        self,
        drug: str = None,
        disease: str = None,
        target: str = None,
        query: DrugDiscoveryQuery = None,
        num_hypotheses: int = 5,
        min_confidence: float = 0.3,
    ) -> HypothesisBatch:
        """
        Generate drug discovery hypotheses.

        Args:
            drug: Drug name
            disease: Disease name
            target: Optional target constraint
            query: Pre-constructed query (alternative to drug/disease)
            num_hypotheses: Number of hypotheses to generate
            min_confidence: Minimum confidence threshold

        Returns:
            HypothesisBatch with generated hypotheses
        """
        start_time = time.time()

        # Construct query if not provided
        if query is None:
            query = DrugDiscoveryQuery(
                query_type="repurposing" if disease else "target_discovery",
                drug=drug,
                disease=disease,
                target=target,
            )

        # Retrieve causal evidence
        logger.info(f"Retrieving causal evidence for {drug} -> {disease}")
        retrieval_result = self.retrieval_engine.retrieve(query)

        # Generate hypotheses using LLM
        raw_hypotheses = self._generate_with_llm(
            retrieval_result,
            num_hypotheses,
        )

        # Calculate uncertainty for each hypothesis
        hypotheses = []
        for raw in raw_hypotheses:
            hypothesis = self._create_hypothesis(
                raw,
                retrieval_result,
            )
            if hypothesis.confidence >= min_confidence:
                hypotheses.append(hypothesis)

        # Sort by confidence
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)

        generation_time = time.time() - start_time

        return HypothesisBatch(
            query=query.__dict__,
            hypotheses=hypotheses,
            generation_time=generation_time,
            retrieval_stats={
                "num_paths": len(retrieval_result.causal_paths),
                "num_documents": len(retrieval_result.supporting_documents),
                "uncertainty_bounds": retrieval_result.uncertainty_bounds,
            },
        )

    def _generate_with_llm(
        self,
        retrieval_result: RetrievalResult,
        num_hypotheses: int,
    ) -> list[dict]:
        """Generate hypotheses using the LLM."""
        # Format causal paths
        causal_paths_text = self._format_causal_paths(
            retrieval_result.causal_paths[:20]
        )

        # Format documents
        documents_text = self._format_documents(
            retrieval_result.supporting_documents[:10]
        )

        # Format evidence statistics
        aggregated = retrieval_result.aggregated_evidence
        key_entities = ", ".join(list(aggregated.get("key_entities", {}).keys())[:10])
        consensus = "; ".join([
            m[0] for m in aggregated.get("consensus_mechanisms", [])[:3]
        ])

        # Create prompt
        prompt = self.GENERATION_PROMPT.format(
            num_hypotheses=num_hypotheses,
            drug=retrieval_result.query.drug or "the drug",
            disease=retrieval_result.query.disease or "the disease",
            causal_paths=causal_paths_text,
            documents=documents_text,
            num_paths=len(retrieval_result.causal_paths),
            num_docs=len(retrieval_result.supporting_documents),
            key_entities=key_entities,
            consensus=consensus,
        )

        # Call LLM
        client = self._get_client()

        try:
            if self.llm_provider == "openai":
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                result = json.loads(response.choices[0].message.content)
            else:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=[
                        {"role": "user", "content": f"{self.SYSTEM_PROMPT}\n\n{prompt}"},
                    ],
                )
                result = json.loads(response.content[0].text)

            return result.get("hypotheses", [])

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return []

    def _format_causal_paths(self, paths: list[CausalPath]) -> str:
        """Format causal paths for the prompt."""
        formatted = []

        for i, path in enumerate(paths, 1):
            mechanism = path.get_mechanism_description()
            confidence = path.get_path_confidence()
            formatted.append(
                f"{i}. {mechanism}\n"
                f"   Confidence: {confidence:.2f}, Length: {path.length}"
            )

        return "\n".join(formatted) if formatted else "No causal paths found."

    def _format_documents(self, documents: list[dict]) -> str:
        """Format supporting documents for the prompt."""
        formatted = []

        for i, doc in enumerate(documents, 1):
            text = doc.get("text", "")[:500]
            source = doc.get("id", "Unknown")
            formatted.append(f"{i}. [{source}] {text}...")

        return "\n".join(formatted) if formatted else "No supporting documents."

    def _create_hypothesis(
        self,
        raw: dict,
        retrieval_result: RetrievalResult,
    ) -> DrugDiscoveryHypothesis:
        """Create a structured hypothesis with uncertainty."""
        # Map confidence string to float
        confidence_map = {"high": 0.85, "medium": 0.65, "low": 0.45}
        base_confidence = confidence_map.get(
            raw.get("confidence", "medium").lower(), 0.65
        )

        # Map novelty string to float
        novelty_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
        novelty = novelty_map.get(raw.get("novelty", "medium").lower(), 0.6)

        # Adjust confidence based on retrieval evidence
        adjusted_confidence = self.uncertainty_quantifier.adjust_confidence(
            base_confidence,
            retrieval_result,
        )

        # Calculate confidence interval
        ci = self.uncertainty_quantifier.calculate_confidence_interval(
            adjusted_confidence,
            retrieval_result.causal_paths,
        )

        return DrugDiscoveryHypothesis(
            drug=raw.get("drug", retrieval_result.query.drug),
            target=raw.get("target", "Unknown"),
            disease=raw.get("disease", retrieval_result.query.disease),
            mechanism=raw.get("mechanism", ""),
            confidence=adjusted_confidence,
            confidence_interval=ci,
            novelty_score=novelty,
            evidence_summary=raw.get("evidence_summary", []),
            supporting_paths=[
                p.to_dict() for p in retrieval_result.causal_paths[:5]
            ],
            num_supporting_sources=len(retrieval_result.supporting_documents),
            suggested_experiments=raw.get("suggested_experiments", []),
            generation_method=f"TC-DrugRAG-{self.model}",
            metadata={
                "limitations": raw.get("limitations", []),
            },
        )

    def generate_from_paths(
        self,
        paths: list[CausalPath],
        drug: str,
        disease: str,
    ) -> list[DrugDiscoveryHypothesis]:
        """
        Generate hypotheses directly from causal paths.

        Useful for testing or when you have pre-computed paths.
        """
        # Create mock retrieval result
        from ..retrieval.causal_retriever import RetrievalResult

        retrieval_result = RetrievalResult(
            query=DrugDiscoveryQuery(
                query_type="repurposing",
                drug=drug,
                disease=disease,
            ),
            causal_paths=paths,
            supporting_documents=[],
            aggregated_evidence={},
            confidence_scores={},
            uncertainty_bounds=(0.0, 1.0),
        )

        raw_hypotheses = self._generate_with_llm(retrieval_result, 3)

        return [
            self._create_hypothesis(raw, retrieval_result)
            for raw in raw_hypotheses
        ]
