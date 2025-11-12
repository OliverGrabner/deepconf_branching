"""
Confidence Peak Branching Manager with Multi-Stage Support

This module implements a novel branching strategy that:
1. Generates initial traces independently
2. Identifies high-confidence peaks within traces
3. Spawns new reasoning paths from those peak points
4. Supports multi-stage branching with doubling strategy
5. Prevents duplicate peaks using exclusion zones

Key features:
- Multi-stage branching (doubles at each stage)
- Branches from WITHIN traces at confidence peaks
- Can branch from branches (deeper genealogy)
- Exclusion zones prevent duplicate peaks
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
    stage: int  # Which branching stage (0=initial, 1+ for branches)
    timestamp: float


@dataclass
class PeakTrace:
    """Enhanced trace with peak branching metadata"""
    trace_idx: int
    stage: int  # 0 for initial, 1+ for branch stages
    depth: int  # Kept for compatibility, same as stage
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
    Manages confidence peak-based branching for self-consistency with multi-stage support

    Key responsibilities:
    - Detect confidence peaks in completed traces
    - Rank peaks globally across all traces
    - Manage multi-stage branching with doubling
    - Prevent duplicate peaks using exclusion zones
    - Track genealogy and token accounting
    """

    def __init__(
        self,
        initial_traces: int = 8,
        max_traces: int = 64,
        confidence_threshold: float = 1.5,
        window_size: int = 512,
        min_peak_distance: int = 256,
        peak_selection_ratio: float = 0.8,
        exclusion_zone_size: int = 200
    ):
        """
        Initialize peak branching manager with multi-stage support

        Args:
            initial_traces: Number of initial independent traces
            max_traces: Maximum total traces (stops when next doubling would exceed)
            confidence_threshold: Minimum confidence for a peak
            window_size: Size of sliding window for confidence calculation
            min_peak_distance: Minimum tokens between peaks in same trace
            peak_selection_ratio: Fraction of trace where peaks are valid (0.8 = middle 80%)
            exclusion_zone_size: Size of exclusion zone around used peaks
        """
        self.initial_traces = initial_traces
        self.max_traces = max_traces
        self.confidence_threshold = confidence_threshold
        self.window_size = window_size
        self.min_peak_distance = min_peak_distance
        self.peak_selection_ratio = peak_selection_ratio
        self.exclusion_zone_size = exclusion_zone_size

        # Calculate branching stages (doubling strategy)
        self.branching_stages = self.calculate_branching_stages()
        self.current_stage = 0

        # State tracking
        self.traces: List[PeakTrace] = []
        self.all_peaks: List[ConfidencePeak] = []
        self.branches: List[PeakBranch] = []
        self.next_trace_idx = 0

        # Exclusion zones to prevent duplicate peaks
        self.used_peak_zones: List[Tuple[int, int, int]] = []  # (trace_idx, start, end)

        print(f"\n=== Multi-Stage Peak Branching Manager Initialized ===")
        print(f"🚀 PEAK DETECTION MODE: ACCELERATION-BASED 🚀")
        print(f"   (Finding moments where confidence accelerates upward)")
        print(f"Initial traces: {initial_traces}")
        print(f"Max traces: {max_traces}")
        print(f"Branching stages: {self.branching_stages}")
        total_after_stages = initial_traces + sum(self.branching_stages)
        print(f"Total traces after all stages: {total_after_stages}")
        print(f"Acceleration threshold: {confidence_threshold} (min positive accel)")
        print(f"Window size: {window_size} tokens")
        print(f"Min peak distance: {min_peak_distance} tokens")
        print(f"Exclusion zone size: {exclusion_zone_size} tokens")
        print(f"Peak selection range: {peak_selection_ratio*100:.0f}% of trace")
        print("="*50)

    def calculate_branching_stages(self) -> List[int]:
        """
        Calculate branching stages using doubling strategy

        Examples:
        - initial=8, max=32: stages=[8, 16] → 8+8+16=32
        - initial=8, max=64: stages=[8, 16, 32] → 8+8+16+32=64
        - initial=4, max=32: stages=[4, 8, 16] → 4+4+8+16=32
        """
        stages = []
        current_total = self.initial_traces
        next_branches = self.initial_traces  # Start by adding same as initial

        while current_total + next_branches <= self.max_traces:
            stages.append(next_branches)
            current_total += next_branches
            next_branches *= 2  # Double for next stage

        return stages

    def create_initial_trace(
        self,
        text: str,
        token_ids: List[int],
        confs: List[float],
        extracted_answer: Optional[str] = None
    ) -> PeakTrace:
        """Create an initial trace (stage 0)"""
        trace = PeakTrace(
            trace_idx=self.next_trace_idx,
            stage=0,
            depth=0,  # For compatibility
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

    def is_position_in_exclusion_zone(self, trace_idx: int, position: int) -> bool:
        """Check if a position is in any exclusion zone"""
        for zone_trace_idx, zone_start, zone_end in self.used_peak_zones:
            if trace_idx == zone_trace_idx and zone_start <= position <= zone_end:
                return True
        return False

    def mark_exclusion_zone(self, trace_idx: int, position: int):
        """Mark an exclusion zone around a used peak"""
        zone_start = max(0, position - self.exclusion_zone_size // 2)
        zone_end = position + self.exclusion_zone_size // 2
        self.used_peak_zones.append((trace_idx, zone_start, zone_end))

    def find_confidence_peaks(self, trace: PeakTrace, check_exclusions: bool = False) -> List[ConfidencePeak]:
        """
        Find confidence peaks in a trace using acceleration detection

        Instead of finding absolute confidence peaks, this finds moments where
        confidence is accelerating upward most rapidly (biggest positive change in velocity).

        Args:
            trace: Trace to analyze
            check_exclusions: Whether to check exclusion zones

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

        # Calculate velocity (1st derivative)
        velocity = np.diff(window_avgs)

        # Calculate acceleration (2nd derivative)
        acceleration = np.diff(velocity)

        # Find peaks where acceleration is highest (biggest positive acceleration)
        # These are inflection points where confidence starts increasing rapidly
        for i in range(1, len(acceleration) - 1):
            # Check if this is a local maximum in acceleration
            # AND acceleration is positive (confidence increasing)
            if (acceleration[i] > acceleration[i-1] and
                acceleration[i] > acceleration[i+1] and
                acceleration[i] > 0.01):  # Minimum acceleration threshold

                # Adjust position accounting for double-diff (lost 2 indices)
                position = i + 2 + self.window_size // 2  # Offset for derivatives

                # Clamp position to valid range
                if position >= len(trace.confs):
                    continue

                trace_fraction = position / len(trace.confs)

                min_fraction = (1 - self.peak_selection_ratio) / 2
                max_fraction = 1 - min_fraction

                if min_fraction <= trace_fraction <= max_fraction:
                    # Check exclusion zones if required
                    if check_exclusions and self.is_position_in_exclusion_zone(trace.trace_idx, position):
                        continue

                    # Ensure minimum distance from other peaks in this trace
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

                        # Use actual confidence at this position (not acceleration value)
                        actual_confidence = confs[position] if position < len(confs) else confs[-1]

                        peak = ConfidencePeak(
                            trace_idx=trace.trace_idx,
                            position=position,
                            confidence=actual_confidence,  # Use actual confidence
                            window_avg=acceleration[i],  # Store acceleration as window_avg for debugging
                            text_snippet=text_snippet
                        )
                        peaks.append(peak)

        return peaks

    def analyze_all_traces_for_stage(self, stage: int) -> List[ConfidencePeak]:
        """
        Analyze all current traces to find peaks for a given stage

        Args:
            stage: Current branching stage (0 for initial analysis)

        Returns:
            Sorted list of valid peaks (highest confidence first)
        """
        print(f"\n🔍 Analyzing {len(self.traces)} traces for ACCELERATION peaks (Stage {stage})...")

        all_peaks = []
        for trace in self.traces:
            # Find peaks, checking exclusions for stages > 0
            peaks = self.find_confidence_peaks(trace, check_exclusions=(stage > 0))

            if stage == 0:
                # Store peaks in trace for initial analysis
                trace.confidence_peaks = peaks

            all_peaks.extend(peaks)

            if peaks:
                peak_positions = [p.position for p in peaks]
                print(f"  Trace {trace.trace_idx} (stage {trace.stage}): Found {len(peaks)} acceleration peaks at positions: {peak_positions[:5]}{'...' if len(peaks) > 5 else ''}")

        # Sort by acceleration value (stored in window_avg)
        all_peaks.sort(key=lambda p: p.window_avg, reverse=True)

        print(f"Total acceleration peaks found: {len(all_peaks)}")
        if all_peaks:
            print(f"Top acceleration value: {all_peaks[0].window_avg:.6f} (at pos {all_peaks[0].position}, conf={all_peaks[0].confidence:.3f})")
            print(f"Note: Peaks sorted by ACCELERATION, not absolute confidence")

        return all_peaks

    def select_peaks_for_stage(self, peaks: List[ConfidencePeak], num_branches: int) -> List[ConfidencePeak]:
        """
        Select top peaks for branching in current stage

        Args:
            peaks: Available peaks
            num_branches: Number of branches to create

        Returns:
            Selected peaks for branching
        """
        # Take top peaks up to num_branches
        selected = peaks[:num_branches]

        print(f"\nSelected {len(selected)} peaks for branching:")
        for i, peak in enumerate(selected[:5]):  # Show first 5
            parent_trace = self.traces[peak.trace_idx]
            print(f"  {i+1}. Trace {peak.trace_idx} (stage {parent_trace.stage}) @ position {peak.position} (conf: {peak.confidence:.3f})")

        if len(selected) > 5:
            print(f"  ... and {len(selected)-5} more")

        return selected

    def prepare_branch_prompts(self, selected_peaks: List[ConfidencePeak]) -> List[Dict[str, Any]]:
        """
        Prepare prompts for branching from selected peaks

        Returns:
            List of dicts with branch information
        """
        branch_prompts = []

        for peak in selected_peaks:
            parent_trace = None
            for trace in self.traces:
                if trace.trace_idx == peak.trace_idx:
                    parent_trace = trace
                    break

            if parent_trace is None:
                continue

            # Extract prefix tokens up to peak position
            prefix_tokens = parent_trace.get_prefix_tokens(peak.position)

            branch_info = {
                'prompt_tokens': prefix_tokens,
                'parent_trace_idx': peak.trace_idx,
                'parent_stage': parent_trace.stage,
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
        stage: int,
        new_text: str,
        new_token_ids: List[int],
        new_confs: List[float],
        extracted_answer: Optional[str] = None
    ) -> PeakTrace:
        """
        Create a branch trace for a given stage

        Args:
            parent_trace_idx: Index of parent trace
            branch_point: Token position where branched
            stage: Current branching stage (1+)
            new_text: Complete text (including inherited prefix)
            new_token_ids: Complete token IDs
            new_confs: Complete confidence scores
            extracted_answer: Extracted answer from branch
        """
        parent_trace = None
        for trace in self.traces:
            if trace.trace_idx == parent_trace_idx:
                parent_trace = trace
                break

        if parent_trace is None:
            raise ValueError(f"Parent trace {parent_trace_idx} not found")

        # Calculate token metrics
        tokens_generated = max(0, len(new_token_ids) - branch_point)  # Only NEW tokens, prevent negative
        total_tokens = len(new_token_ids)

        # Warn if branch was truncated (branch_point beyond generated tokens)
        if branch_point > len(new_token_ids):
            print(f"  ⚠️  Branch truncated: branch_point {branch_point} > total tokens {len(new_token_ids)}")

        trace = PeakTrace(
            trace_idx=self.next_trace_idx,
            stage=stage,
            depth=stage,  # For compatibility
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
            stage=stage,
            timestamp=time.time()
        )
        self.branches.append(branch_event)

        self.traces.append(trace)
        self.next_trace_idx += 1
        return trace

    def run_branching_stage(self, stage_idx: int, num_branches: int) -> List[ConfidencePeak]:
        """
        Run a single branching stage

        Args:
            stage_idx: Stage index (0-based)
            num_branches: Number of branches to create in this stage

        Returns:
            List of selected peaks for external branch generation
        """
        stage_num = stage_idx + 1  # Human-readable stage number

        print(f"\n{'='*60}")
        print(f"STAGE {stage_num}: Creating {num_branches} branches")
        print(f"Current traces: {len(self.traces)}")
        print(f"{'='*60}")

        # Find peaks in all current traces
        peaks = self.analyze_all_traces_for_stage(stage_num)

        if not peaks:
            print(f"WARNING: No valid peaks found for stage {stage_num}")
            return []

        # Select top peaks for this stage
        selected_peaks = self.select_peaks_for_stage(peaks, min(num_branches, len(peaks)))

        # Mark exclusion zones for selected peaks
        for peak in selected_peaks:
            self.mark_exclusion_zone(peak.trace_idx, peak.position)

        print(f"Marked {len(selected_peaks)} exclusion zones")

        return selected_peaks

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the branching process"""
        # Group traces by stage
        traces_by_stage = {}
        for trace in self.traces:
            stage = trace.stage
            if stage not in traces_by_stage:
                traces_by_stage[stage] = []
            traces_by_stage[stage].append(trace)

        # Calculate token savings from prefix caching
        total_tokens_generated = sum(t.tokens_generated for t in self.traces)
        total_tokens_including_prefix = sum(t.total_tokens for t in self.traces)
        prefix_savings = total_tokens_including_prefix - total_tokens_generated

        # Stage-specific stats
        stage_stats = {}
        for stage, stage_traces in traces_by_stage.items():
            stage_stats[f'stage_{stage}_traces'] = len(stage_traces)
            stage_stats[f'stage_{stage}_avg_tokens'] = np.mean([t.tokens_generated for t in stage_traces]) if stage_traces else 0

        stats = {
            'total_traces': len(self.traces),
            'initial_traces': len(traces_by_stage.get(0, [])),
            'total_branch_traces': len(self.traces) - len(traces_by_stage.get(0, [])),
            'num_stages': len(self.branching_stages),
            'branching_stages': self.branching_stages,
            'traces_by_stage': {k: len(v) for k, v in traces_by_stage.items()},
            'total_peaks_found': len(self.all_peaks),
            'branches_created': len(self.branches),
            'exclusion_zones': len(self.used_peak_zones),
            'total_tokens_generated': total_tokens_generated,
            'total_tokens_with_prefix': total_tokens_including_prefix,
            'prefix_cache_savings': prefix_savings,
            'prefix_cache_savings_pct': (prefix_savings / total_tokens_including_prefix * 100) if total_tokens_including_prefix > 0 else 0,
            'avg_branch_point': np.mean([b.branch_point for b in self.branches]) if self.branches else 0,
        }

        # Add stage-specific stats
        stats.update(stage_stats)

        return stats

    def print_summary(self):
        """Print comprehensive summary of multi-stage branching results"""
        stats = self.get_statistics()

        print("\n" + "="*70)
        print("MULTI-STAGE PEAK BRANCHING SUMMARY")
        print("="*70)

        # Trace distribution
        print(f"Total traces: {stats['total_traces']}")
        print(f"  Initial: {stats['initial_traces']}")
        for stage in range(1, stats['num_stages'] + 1):
            count = stats['traces_by_stage'].get(stage, 0)
            if count > 0:
                print(f"  Stage {stage}: {count} branches")

        print(f"\nBranching stages executed: {self.branching_stages}")
        print(f"Exclusion zones created: {stats['exclusion_zones']}")

        print(f"\nToken Usage:")
        print(f"  Total generated: {stats['total_tokens_generated']:,}")
        print(f"  With prefix reuse: {stats['total_tokens_with_prefix']:,}")
        print(f"  Prefix cache savings: {stats['prefix_cache_savings']:,} ({stats['prefix_cache_savings_pct']:.1f}%)")

        # Per-stage token averages
        print(f"\nAverage tokens per stage:")
        for stage in range(stats['num_stages'] + 1):
            key = f'stage_{stage}_avg_tokens'
            if key in stats:
                print(f"  Stage {stage}: {stats[key]:.0f} tokens")

        print(f"\nAvg branch point: {stats['avg_branch_point']:.0f} tokens into trace")
        print("="*70)