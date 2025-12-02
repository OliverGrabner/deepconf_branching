"""
Output classes for SCLLM
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class SCOutput:
    """Output container for self-consistency results"""

    # Primary results
    final_answer: Optional[str] = None
    voted_answer: Optional[str] = None

    # Traces and voting
    all_traces: List[Dict[str, Any]] = field(default_factory=list)
    voting_answers: List[str] = field(default_factory=list)

    # Statistics
    total_traces_count: int = 0

    # Token statistics
    total_tokens: int = 0
    total_tokens_generated: int = 0  # Only newly generated tokens (excludes inherited from branching)
    avg_tokens_per_trace: float = 0.0
    avg_tokens_generated_per_trace: float = 0.0

    # Timing information
    tokenizer_init_time: float = 0.0
    llm_init_time: float = 0.0
    generation_time: float = 0.0
    processing_time: float = 0.0
    total_time: float = 0.0

    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)

    # Branching-specific information
    branch_events: List[Dict[str, Any]] = field(default_factory=list)
    branch_genealogy: Dict[str, Any] = field(default_factory=dict)
    branching_config: Dict[str, Any] = field(default_factory=dict)

    # Peak branching specific
    peak_branching_config: Dict[str, Any] = field(default_factory=dict)
    peak_branching_stats: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    mode: str = "offline"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "final_answer": self.final_answer,
            "voted_answer": self.voted_answer,
            "all_traces": self.all_traces,
            "voting_answers": self.voting_answers,
            "total_traces_count": self.total_traces_count,
            "token_stats": {
                "total_tokens": self.total_tokens,
                "total_tokens_generated": self.total_tokens_generated,
                "avg_tokens_per_trace": self.avg_tokens_per_trace,
                "avg_tokens_generated_per_trace": self.avg_tokens_generated_per_trace,
            },
            "timing_stats": {
                "tokenizer_init_time": self.tokenizer_init_time,
                "llm_init_time": self.llm_init_time,
                "generation_time": self.generation_time,
                "processing_time": self.processing_time,
                "total_time": self.total_time,
            },
            "config": self.config,
            "mode": self.mode,
            "timestamp": self.timestamp,
            "branch_events": self.branch_events,
            "branch_genealogy": self.branch_genealogy,
            "branching_config": self.branching_config,
            "peak_branching_config": self.peak_branching_config,
            "peak_branching_stats": self.peak_branching_stats,
        }

    def print_summary(self):
        """Print a formatted summary of the results"""
        print(f"\n=== SC Summary ===")
        print(f"Mode: {self.mode}")
        print(f"Generated traces: {self.total_traces_count}")

        if self.mode == "branching" and self.branch_genealogy:
            stats = self.branch_genealogy.get('statistics', {})
            print(f"Original traces: {stats.get('original_traces', 0)}")
            print(f"Branched traces: {stats.get('branched_traces', 0)}")
            print(f"Total branch events: {stats.get('total_branch_events', 0)}")

        print(f"Valid answers for voting: {len(self.voting_answers)}")

        if self.final_answer:
            print(f"Final answer: {self.final_answer}")

        print(f"Total tokens: {self.total_tokens}")
        if self.mode == "branching" and self.total_tokens_generated > 0:
            print(f"Tokens generated (excluding inherited): {self.total_tokens_generated}")

        if self.generation_time > 0:
            print(f"Generation time: {self.generation_time:.2f}s")
            print(f"Generation throughput: {self.total_tokens / self.generation_time:.1f} tokens/second")

        print(f"Total time: {self.total_time:.2f}s")

    @property
    def overall_throughput(self) -> float:
        """Overall token generation throughput"""
        if self.generation_time > 0:
            return self.total_tokens / self.generation_time
        return 0.0
