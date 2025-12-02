"""
Branching Self-Consistency Manager

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class BranchEvent:
    """Record of a branching event"""
    iteration: int
    parent_trace_idx: int
    child_trace_idx: int
    branch_point_tokens: int
    parent_tail_confidence: float
    timestamp: float


@dataclass
class TraceState:
    """State of an active trace during generation"""
    trace_idx: int  # Unique identifier
    parent_idx: Optional[int]  # None for original traces
    generation_started_at_iteration: int
    generation_started_at_tokens: int
    current_text: str
    current_token_ids: List[int]
    current_confs: List[float]
    is_complete: bool = False

    def get_tail_confidence(self, tail_window: int) -> float:
        """Compute tail confidence for this trace"""
        if not self.current_confs:
            return 0.0
        tail = self.current_confs[-tail_window:] if len(self.current_confs) > tail_window else self.current_confs
        return float(np.mean(tail)) if tail else 0.0


class BranchingManager:
    """
    Manages dynamic trace branching during self-consistency generation

    Key responsibilities:
    - Maintain branching schedule (when to branch, how many)
    - Select high-confidence traces for branching
    - Track genealogy (parent-child relationships)
    - Monitor progress toward max_traces goal
    """

    def __init__(
        self,
        start_traces: int = 8,
        max_traces: int = 32,
        selected_percent: float = 0.60,
        n_iterations: int = 10,
        branch_goal: float = 0.75,
        average_tokens: int = 8000,
        tail_window: int = 2048
    ):
        """
        Initialize branching manager

        Args:
            start_traces: Number of initial traces
            max_traces: Maximum allowed traces
            selected_percent: Top % of traces eligible for branching (e.g., 0.60 = top 60%)
            n_iterations: Number of check points for branching decisions
            branch_goal: Target completion percentage for branching (e.g., 0.75 = 75%)
            average_tokens: Historical average tokens for this problem
            tail_window: Number of tokens for tail confidence computation
        """
        self.start_traces = start_traces
        self.max_traces = max_traces
        self.selected_percent = selected_percent
        self.n_iterations = n_iterations
        self.branch_goal = branch_goal
        self.average_tokens = average_tokens
        self.tail_window = tail_window

        # Calculate branching schedule
        self.branch_deadline_tokens = int(branch_goal * average_tokens)
        self.stride = int(self.branch_deadline_tokens / n_iterations)
        self.branches_needed = max_traces - start_traces
        self.branches_per_iteration = int(np.ceil(self.branches_needed / n_iterations))

        # State tracking
        self.active_traces: List[TraceState] = []
        self.branch_events: List[BranchEvent] = []
        self.current_iteration = 0
        self.next_trace_idx = 0  # Counter for assigning unique trace IDs

        print(f"\n=== Branching Manager Initialized ===")
        print(f"Start traces: {start_traces}, Max traces: {max_traces}")
        print(f"Average tokens: {average_tokens}, Branch goal: {branch_goal*100:.0f}%")
        print(f"Stride: {self.stride} tokens, Iterations: {n_iterations}")
        print(f"Branches needed: {self.branches_needed}, Per iteration: {self.branches_per_iteration}")
        print(f"Branch deadline: {self.branch_deadline_tokens} tokens")
        print(f"Top {selected_percent*100:.0f}% eligible for branching")
        print("="*40)

    def initialize_traces(self, num_traces: int) -> List[TraceState]:
        """Create initial trace states"""
        traces = []
        for i in range(num_traces):
            trace = TraceState(
                trace_idx=self.next_trace_idx,
                parent_idx=None,
                generation_started_at_iteration=0,
                generation_started_at_tokens=0,
                current_text="",
                current_token_ids=[],
                current_confs=[]
            )
            traces.append(trace)
            self.next_trace_idx += 1

        self.active_traces = traces
        return traces

    def should_branch(self) -> bool:
        """Check if we should create new branches at this iteration"""
        current_count = len(self.active_traces)
        current_tokens = self.current_iteration * self.stride

        # Don't branch if:
        # 1. Already at max capacity
        # 2. Past the branching deadline
        if current_count >= self.max_traces:
            return False
        if current_tokens >= self.branch_deadline_tokens:
            return False

        return True

    def select_branch_candidates(self) -> List[Tuple[int, float]]:
        """
        Select traces eligible for branching based on tail confidence

        Returns:
            List of (trace_idx, tail_confidence) tuples for top percent of traces
        """
        # Compute tail confidence for all active traces
        traces_with_conf = []
        for trace in self.active_traces:
            tail_conf = trace.get_tail_confidence(self.tail_window)
            traces_with_conf.append((trace.trace_idx, tail_conf))

        # Sort by confidence (descending)
        sorted_traces = sorted(traces_with_conf, key=lambda x: x[1], reverse=True)

        # Select top percent
        n_candidates = max(1, int(len(sorted_traces) * self.selected_percent))
        candidates = sorted_traces[:n_candidates]

        return candidates

    def select_branches_to_create(self, candidates: List[Tuple[int, float]]) -> List[int]:
        """
        Uniformly sample from candidates to decide which traces to branch

        Args:
            candidates: List of (trace_idx, confidence) tuples

        Returns:
            List of trace indices to branch (may contain duplicates)
        """
        if not candidates:
            return []

        # Calculate how many branches to create
        current_count = len(self.active_traces)
        branches_to_create = min(
            self.branches_per_iteration,
            self.max_traces - current_count
        )

        if branches_to_create <= 0:
            return []

        # Uniformly sample from candidates (with replacement)
        candidate_indices = [idx for idx, _ in candidates]
        selected = np.random.choice(
            candidate_indices,
            size=min(branches_to_create, len(candidate_indices)),
            replace=True  # Allow same trace to be branched multiple times
        )

        return selected.tolist()

    def create_branches(
        self,
        parent_indices: List[int],
        timestamp: float
    ) -> List[TraceState]:
        """
        Create new branch traces from selected parents

        Args:
            parent_indices: List of parent trace indices to branch from
            timestamp: Current timestamp for logging

        Returns:
            List of newly created TraceState objects
        """
        new_traces = []
        current_tokens = self.current_iteration * self.stride

        # Get mapping from trace_idx to trace object
        trace_map = {trace.trace_idx: trace for trace in self.active_traces}

        for parent_idx in parent_indices:
            if parent_idx not in trace_map:
                print(f"Warning: Parent trace {parent_idx} not found, skipping")
                continue

            parent_trace = trace_map[parent_idx]

            # Create child trace by copying parent's state
            child_trace = TraceState(
                trace_idx=self.next_trace_idx,
                parent_idx=parent_idx,
                generation_started_at_iteration=self.current_iteration,
                generation_started_at_tokens=current_tokens,
                current_text=parent_trace.current_text,
                current_token_ids=parent_trace.current_token_ids.copy(),
                current_confs=parent_trace.current_confs.copy()
            )

            # Record branch event
            branch_event = BranchEvent(
                iteration=self.current_iteration,
                parent_trace_idx=parent_idx,
                child_trace_idx=self.next_trace_idx,
                branch_point_tokens=current_tokens,
                parent_tail_confidence=parent_trace.get_tail_confidence(self.tail_window),
                timestamp=timestamp
            )

            self.branch_events.append(branch_event)
            new_traces.append(child_trace)

            self.next_trace_idx += 1

        # Add new traces to active pool
        self.active_traces.extend(new_traces)

        return new_traces

    def get_genealogy(self) -> Dict[str, Any]:
        """
        Get complete genealogy information

        Returns:
            Dictionary with:
            - tree: parent-child relationships
            - events: chronological branching events
            - statistics: summary stats
        """
        # Build parent-child tree
        tree = {}
        for trace in self.active_traces:
            if trace.parent_idx is None:
                tree[trace.trace_idx] = {'parent': None, 'children': []}
            else:
                tree[trace.trace_idx] = {'parent': trace.parent_idx, 'children': []}

        # Add children references
        for trace in self.active_traces:
            if trace.parent_idx is not None and trace.parent_idx in tree:
                tree[trace.parent_idx]['children'].append(trace.trace_idx)

        # Compile events
        events = [
            {
                'iteration': e.iteration,
                'parent_trace_idx': e.parent_trace_idx,
                'child_trace_idx': e.child_trace_idx,
                'branch_point_tokens': e.branch_point_tokens,
                'parent_tail_confidence': e.parent_tail_confidence,
                'timestamp': e.timestamp
            }
            for e in self.branch_events
        ]

        # Calculate statistics
        original_traces = [t for t in self.active_traces if t.parent_idx is None]
        branched_traces = [t for t in self.active_traces if t.parent_idx is not None]

        stats = {
            'total_traces': len(self.active_traces),
            'original_traces': len(original_traces),
            'branched_traces': len(branched_traces),
            'total_branch_events': len(self.branch_events),
            'iterations_completed': self.current_iteration,
            'final_tokens': self.current_iteration * self.stride
        }

        return {
            'tree': tree,
            'events': events,
            'statistics': stats
        }

    def advance_iteration(self):
        """Move to next iteration"""
        self.current_iteration += 1

    def print_status(self):
        """Print current branching status"""
        current_tokens = self.current_iteration * self.stride
        print(f"\n[Iteration {self.current_iteration}] Tokens: {current_tokens}/{self.average_tokens}")
        print(f"Active traces: {len(self.active_traces)}/{self.max_traces}")
        print(f"Branches created so far: {len(self.branch_events)}")
