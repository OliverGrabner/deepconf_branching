"""
Confidence Peak Branching Manager

This module implements a novel branching strategy that:
1. Generates initial traces independently
2. Identifies high-confidence peaks within traces
3. Spawns new reasoning paths from those peak points

Key difference from traditional branching:
- Branches from WITHIN traces at confidence peaks (not from the end)
- Single branching round after initial generation
- Leverages vLLM prefix caching for efficiency

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import time


@dataclass
class ConfidencePeak:
    """Record of a confidence peak in a trace"""
    trace_idx: int
    position: int  # Token position in trace
    confidence: float
    window_avg: float
    text_snippet: str  # Text around the peak for context


@dataclass
class PeakBranch:
    """Record of a branch created from a confidence peak"""
    branch_idx: int  # New trace index
    parent_idx: int  # Original trace index
    branch_point: int  # Token position where branched
    parent_confidence: float  # Confidence at branch point
    depth: int  # 0 for initial, 1 for branches
    timestamp: float


@dataclass
class PeakTrace:
    """Enhanced trace with peak branching metadata"""
    trace_idx: int
    depth: int  # 0 for initial traces, 1 for branches
    parent_idx: Optional[int]  # None for initial traces
    branch_point_tokens: Optional[int]  # Where this branched from parent

    # Content
    text: str
    token_ids: List[int]
    confs: List[float]

    # Metrics
    tokens_generated: int  # NEW tokens only (excluding prefix)
    total_tokens: int  # Total including prefix
    extracted_answer: Optional[str] = None

    # Peaks found in this trace
    confidence_peaks: List[ConfidencePeak] = field(default_factory=list)

    def get_prefix_tokens(self, up_to_position: int) -> List[int]:
        """Get token prefix up to specified position"""
        return self.token_ids[:up_to_position]

    def get_prefix_text(self, up_to_position: int) -> str:
        """Get text prefix up to specified position (approximate)"""
        # This is approximate - exact text reconstruction would need tokenizer
        ratio = up_to_position / len(self.token_ids) if self.token_ids else 0
        char_position = int(len(self.text) * ratio)
        return self.text[:char_position]


class PeakBranchingManager:
    """
    Manages confidence peak-based branching for self-consistency

    Key responsibilities:
    - Detect confidence peaks in completed traces
    - Rank peaks globally across all traces
    - Manage branching from high-confidence points
    - Track genealogy and token accounting
    """

    def __init__(
        self,
        initial_traces: int = 8,
        total_traces: int = 32,
        confidence_threshold: float = 1.5,
        window_size: int = 512,
        min_peak_distance: int = 256,
        peak_selection_ratio: float = 0.8
    ):
        """
        Initialize peak branching manager

        Args:
            initial_traces: Number of initial independent traces
            total_traces: Total traces after branching
            confidence_threshold: Minimum confidence for a peak
            window_size: Size of sliding window for confidence calculation
            min_peak_distance: Minimum tokens between peaks
            peak_selection_ratio: Fraction of peak position (0.2-0.8) for valid peaks
        """
        self.initial_traces = initial_traces
        self.total_traces = total_traces
        self.confidence_threshold = confidence_threshold
        self.window_size = window_size
        self.min_peak_distance = min_peak_distance
        self.peak_selection_ratio = peak_selection_ratio

        # Calculate branching parameters
        self.branches_to_create = total_traces - initial_traces

        # State tracking
        self.traces: List[PeakTrace] = []
        self.all_peaks: List[ConfidencePeak] = []
        self.branches: List[PeakBranch] = []
        self.next_trace_idx = 0

        print(f"\n=== Peak Branching Manager Initialized ===")
        print(f"Initial traces: {initial_traces}")
        print(f"Total traces target: {total_traces}")
        print(f"Branches to create: {self.branches_to_create}")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"Window size: {window_size} tokens")
        print(f"Min peak distance: {min_peak_distance} tokens")
        print(f"Peak selection range: {peak_selection_ratio*100:.0f}% of trace")
        print("="*45)

    def create_initial_trace(
        self,
        text: str,
        token_ids: List[int],
        confs: List[float],
        extracted_answer: Optional[str] = None
    ) -> PeakTrace:
        """Create an initial trace (depth 0)"""
        trace = PeakTrace(
            trace_idx=self.next_trace_idx,
            depth=0,
            parent_idx=None,
            branch_point_tokens=None,
            text=text,
            token_ids=token_ids,
            confs=confs,
            tokens_generated=len(token_ids),
            total_tokens=len(token_ids),
            extracted_answer=extracted_answer
        )
        self.traces.append(trace)
        self.next_trace_idx += 1
        return trace

    def find_confidence_peaks(self, trace: PeakTrace) -> List[ConfidencePeak]:
        """
        Find confidence peaks in a trace using sliding window

        Returns:
            List of ConfidencePeak objects
        """
        if not trace.confs or len(trace.confs) < self.window_size:
            return []

        peaks = []
        confs = np.array(trace.confs)

        # Compute sliding window averages
        window_avgs = []
        for i in range(len(confs) - self.window_size + 1):
            window = confs[i:i + self.window_size]
            window_avgs.append(np.mean(window))

        window_avgs = np.array(window_avgs)

        # Find local maxima above threshold
        for i in range(1, len(window_avgs) - 1):
            # Check if it's a local maximum
            if (window_avgs[i] > window_avgs[i-1] and
                window_avgs[i] > window_avgs[i+1] and
                window_avgs[i] > self.confidence_threshold):

                # Check position is in valid range (20-80% of trace)
                position = i + self.window_size // 2  # Center of window
                trace_fraction = position / len(trace.confs)

                min_fraction = (1 - self.peak_selection_ratio) / 2
                max_fraction = 1 - min_fraction

                if min_fraction <= trace_fraction <= max_fraction:
                    # Ensure minimum distance from other peaks
                    too_close = False
                    for p in peaks:
                        if abs(p.position - position) < self.min_peak_distance:
                            too_close = True
                            break

                    if not too_close:
                        # Extract text snippet around peak
                        text_position = int(len(trace.text) * trace_fraction)
                        snippet_start = max(0, text_position - 50)
                        snippet_end = min(len(trace.text), text_position + 50)
                        text_snippet = trace.text[snippet_start:snippet_end]

                        peak = ConfidencePeak(
                            trace_idx=trace.trace_idx,
                            position=position,
                            confidence=window_avgs[i],
                            window_avg=window_avgs[i],
                            text_snippet=text_snippet
                        )
                        peaks.append(peak)

        return peaks

    def analyze_all_traces(self) -> List[ConfidencePeak]:
        """
        Analyze all initial traces to find and rank confidence peaks

        Returns:
            Sorted list of all peaks (highest confidence first)
        """
        print(f"\nAnalyzing {len(self.traces)} traces for confidence peaks...")

        all_peaks = []
        for trace in self.traces:
            if trace.depth == 0:  # Only analyze initial traces
                peaks = self.find_confidence_peaks(trace)
                trace.confidence_peaks = peaks
                all_peaks.extend(peaks)

                if peaks:
                    print(f"  Trace {trace.trace_idx}: Found {len(peaks)} peaks")
                    for p in peaks[:2]:  # Show top 2
                        print(f"    - Position {p.position}, confidence {p.confidence:.3f}")

        # Sort by confidence (highest first)
        all_peaks.sort(key=lambda p: p.confidence, reverse=True)

        print(f"\nTotal peaks found: {len(all_peaks)}")
        if all_peaks:
            print(f"Top confidence: {all_peaks[0].confidence:.3f}")
            print(f"Peaks above threshold: {sum(1 for p in all_peaks if p.confidence > self.confidence_threshold)}")

        self.all_peaks = all_peaks
        return all_peaks

    def select_peaks_for_branching(self) -> List[ConfidencePeak]:
        """
        Select which peaks to branch from

        Returns:
            List of selected peaks for branching
        """
        # Take top peaks up to branches_to_create
        selected = self.all_peaks[:self.branches_to_create]

        print(f"\nSelected {len(selected)} peaks for branching:")
        for i, peak in enumerate(selected[:5]):  # Show first 5
            print(f"  {i+1}. Trace {peak.trace_idx} @ position {peak.position} (conf: {peak.confidence:.3f})")

        if len(selected) > 5:
            print(f"  ... and {len(selected)-5} more")

        return selected

    def prepare_branch_prompts(self, selected_peaks: List[ConfidencePeak]) -> List[Dict[str, Any]]:
        """
        Prepare prompts for branching from selected peaks

        Returns:
            List of dicts with branch information:
            - 'prompt_tokens': Token IDs to use as prompt
            - 'parent_trace_idx': Original trace index
            - 'branch_point': Token position of branch
            - 'parent_confidence': Confidence at branch point
        """
        branch_prompts = []

        for peak in selected_peaks:
            parent_trace = self.traces[peak.trace_idx]

            # Extract prefix tokens up to peak position
            prefix_tokens = parent_trace.get_prefix_tokens(peak.position)

            branch_info = {
                'prompt_tokens': prefix_tokens,
                'parent_trace_idx': peak.trace_idx,
                'branch_point': peak.position,
                'parent_confidence': peak.confidence,
                'prefix_text_preview': parent_trace.get_prefix_text(peak.position)[-100:]  # Last 100 chars
            }
            branch_prompts.append(branch_info)

        return branch_prompts

    def create_branch_trace(
        self,
        parent_trace_idx: int,
        branch_point: int,
        new_text: str,
        new_token_ids: List[int],
        new_confs: List[float],
        extracted_answer: Optional[str] = None
    ) -> PeakTrace:
        """
        Create a branch trace (depth 1)

        Args:
            parent_trace_idx: Index of parent trace
            branch_point: Token position where branched
            new_text: Complete text (including inherited prefix)
            new_token_ids: Complete token IDs
            new_confs: Complete confidence scores
            extracted_answer: Extracted answer from branch
        """
        parent_trace = self.traces[parent_trace_idx]

        # Calculate token metrics
        tokens_generated = len(new_token_ids) - branch_point  # Only NEW tokens
        total_tokens = len(new_token_ids)

        trace = PeakTrace(
            trace_idx=self.next_trace_idx,
            depth=1,
            parent_idx=parent_trace_idx,
            branch_point_tokens=branch_point,
            text=new_text,
            token_ids=new_token_ids,
            confs=new_confs,
            tokens_generated=tokens_generated,
            total_tokens=total_tokens,
            extracted_answer=extracted_answer
        )

        # Record branch event
        branch_event = PeakBranch(
            branch_idx=self.next_trace_idx,
            parent_idx=parent_trace_idx,
            branch_point=branch_point,
            parent_confidence=parent_trace.confs[branch_point] if branch_point < len(parent_trace.confs) else 0,
            depth=1,
            timestamp=time.time()
        )
        self.branches.append(branch_event)

        self.traces.append(trace)
        self.next_trace_idx += 1
        return trace

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the branching process"""
        initial_traces = [t for t in self.traces if t.depth == 0]
        branch_traces = [t for t in self.traces if t.depth == 1]

        # Calculate token savings from prefix caching
        total_tokens_generated = sum(t.tokens_generated for t in self.traces)
        total_tokens_including_prefix = sum(t.total_tokens for t in self.traces)
        prefix_savings = total_tokens_including_prefix - total_tokens_generated

        stats = {
            'total_traces': len(self.traces),
            'initial_traces': len(initial_traces),
            'branch_traces': len(branch_traces),
            'total_peaks_found': len(self.all_peaks),
            'peaks_above_threshold': sum(1 for p in self.all_peaks if p.confidence > self.confidence_threshold),
            'branches_created': len(self.branches),
            'total_tokens_generated': total_tokens_generated,
            'total_tokens_with_prefix': total_tokens_including_prefix,
            'prefix_cache_savings': prefix_savings,
            'prefix_cache_savings_pct': (prefix_savings / total_tokens_including_prefix * 100) if total_tokens_including_prefix > 0 else 0,
            'avg_tokens_initial': np.mean([t.tokens_generated for t in initial_traces]) if initial_traces else 0,
            'avg_tokens_branch': np.mean([t.tokens_generated for t in branch_traces]) if branch_traces else 0,
            'avg_branch_point': np.mean([b.branch_point for b in self.branches]) if self.branches else 0,
            'avg_peak_confidence': np.mean([p.confidence for p in self.all_peaks]) if self.all_peaks else 0
        }

        return stats

    def print_summary(self):
        """Print summary of branching results"""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("PEAK BRANCHING SUMMARY")
        print("="*60)
        print(f"Total traces: {stats['total_traces']} ({stats['initial_traces']} initial + {stats['branch_traces']} branches)")
        print(f"Peaks found: {stats['total_peaks_found']} ({stats['peaks_above_threshold']} above threshold)")
        print(f"Branches created: {stats['branches_created']}")
        print(f"\nToken Usage:")
        print(f"  Total generated: {stats['total_tokens_generated']:,}")
        print(f"  With prefix reuse: {stats['total_tokens_with_prefix']:,}")
        print(f"  Prefix cache savings: {stats['prefix_cache_savings']:,} ({stats['prefix_cache_savings_pct']:.1f}%)")
        print(f"  Avg initial trace: {stats['avg_tokens_initial']:.0f} tokens")
        print(f"  Avg branch trace: {stats['avg_tokens_branch']:.0f} tokens (new only)")
        print(f"  Avg branch point: {stats['avg_branch_point']:.0f} tokens into trace")
        print(f"  Avg peak confidence: {stats['avg_peak_confidence']:.3f}")
        print("="*60)