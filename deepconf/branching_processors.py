"""
Branching processors for confidence-based trace spawning

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from collections import deque
import random
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)
from vllm import SamplingParams
from vllm.config import VllmConfig


class ConfidenceBranchingProcessor:
    """
    Processor that decides whether to spawn new branches based on confidence levels.
    Higher confidence increases the probability of creating offspring traces.
    """
    
    def __init__(
        self, 
        base_branch_prob: float = 0.1,
        max_branch_prob: float = 0.9,
        confidence_threshold: float = 2.0,
        conf_group_size: int = 128,
        conf_topk: int = 20,
        min_steps_before_branch: int = 50,
        branch_cooldown: int = 100,
        max_branches_per_trace: int = 5
    ) -> None:
        """
        Initialize the branching processor.
        
        Args:
            base_branch_prob: Minimum probability of branching (when confidence is low)
            max_branch_prob: Maximum probability of branching (when confidence is high)
            confidence_threshold: Confidence value that maps to max_branch_prob
            conf_group_size: Size of sliding window for confidence computation
            conf_topk: Top-k tokens to consider for confidence
            min_steps_before_branch: Minimum tokens before allowing first branch
            branch_cooldown: Minimum tokens between branches
            max_branches_per_trace: Maximum number of branches per trace
        """
        self.base_branch_prob = base_branch_prob
        self.max_branch_prob = max_branch_prob
        self.confidence_threshold = confidence_threshold
        self.conf_topk = conf_topk
        self.conf_group_size = conf_group_size
        self.min_steps_before_branch = min_steps_before_branch
        self.branch_cooldown = branch_cooldown
        self.max_branches_per_trace = max_branches_per_trace
        
        # State tracking
        self.conf_group_list = deque(maxlen=conf_group_size)
        self.conf_grouped = 0.0
        self.step_count = 0
        self.last_branch_step = -branch_cooldown
        self.branch_count = 0
        self.branch_requests = []
        
    def compute_conf(self, logits: torch.Tensor) -> float:
        """Compute the confidence score based on the logits"""
        probabilities = torch.softmax(logits, dim=-1)
        top_probs, _ = torch.topk(probabilities, self.conf_topk, dim=-1)
        log_probs = torch.log(top_probs)
        # Higher confidence = lower entropy = lower negative log prob
        return -log_probs.sum().item() / self.conf_topk
    
    def get_branch_probability(self, confidence: float) -> float:
        """
        Map confidence to branching probability.
        Uses a sigmoid-like function to map confidence to probability.
        """
        # Normalize confidence to [0, 1] range
        normalized_conf = min(1.0, confidence / self.confidence_threshold)
        
        # Use a smooth interpolation
        # When confidence is 0 -> base_branch_prob
        # When confidence is confidence_threshold -> max_branch_prob
        branch_prob = self.base_branch_prob + (self.max_branch_prob - self.base_branch_prob) * normalized_conf
        
        return branch_prob
    
    def should_branch(self, avg_confidence: float) -> Tuple[bool, float]:
        """
        Decide whether to spawn a new branch based on current confidence.
        
        Returns:
            (should_branch, branch_probability)
        """
        # Check if we can branch
        if self.step_count < self.min_steps_before_branch:
            return False, 0.0
            
        if self.step_count - self.last_branch_step < self.branch_cooldown:
            return False, 0.0
            
        if self.branch_count >= self.max_branches_per_trace:
            return False, 0.0
        
        # Calculate branch probability based on confidence
        branch_prob = self.get_branch_probability(avg_confidence)
        
        # Random decision
        should_branch = random.random() < branch_prob
        
        if should_branch:
            self.last_branch_step = self.step_count
            self.branch_count += 1
        
        return should_branch, branch_prob
    
    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Process logits and decide on branching.
        
        Returns:
            (processed_logits, branch_info)
        """
        self.step_count += 1
        
        # Compute confidence for current token
        new_conf = self.compute_conf(logits)
        
        # Update sliding window
        if len(self.conf_group_list) < self.conf_group_size:
            self.conf_group_list.append(new_conf)
            self.conf_grouped += new_conf
        else:
            self.conf_grouped -= self.conf_group_list[0]
            self.conf_group_list.append(new_conf)
            self.conf_grouped += new_conf
        
        # Calculate average confidence
        avg_confidence = self.conf_grouped / len(self.conf_group_list) if self.conf_group_list else 0
        
        # Decide on branching
        should_branch, branch_prob = self.should_branch(avg_confidence)
        
        branch_info = None
        if should_branch:
            branch_info = {
                'step': self.step_count,
                'confidence': avg_confidence,
                'branch_probability': branch_prob,
                'output_ids_so_far': output_ids.copy(),
                'branch_count': self.branch_count
            }
            self.branch_requests.append(branch_info)
        
        # Return unmodified logits and branch info
        return logits, branch_info


class WrappedBranchingProcessor(AdapterLogitsProcessor):
    """
    Wrapper that integrates branching processor with vLLM.
    This processor monitors confidence and signals when to spawn new traces.
    """
    
    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ):
        super().__init__(vllm_config, device, is_pin_memory)
        self.is_cuda = device.type == "cuda"
        self.pending_branches = []
    
    def is_argmax_invariant(self) -> bool:
        return True  # This processor doesn't modify logits
    
    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> Optional[RequestLogitsProcessor]:
        """
        Create a new request-level branching processor.
        
        The processor monitors confidence and collects branching requests,
        but doesn't modify logits directly.
        """
        if (
            not self.is_cuda
            or (branching_config := params.extra_args
                and params.extra_args.get("branching_config")
            ) is None
        ):
            return None
        
        config = branching_config
        return ConfidenceBranchingProcessor(
            base_branch_prob=config.get('base_branch_prob', 0.1),
            max_branch_prob=config.get('max_branch_prob', 0.9),
            confidence_threshold=config.get('confidence_threshold', 2.0),
            conf_group_size=config.get('conf_group_size', 128),
            conf_topk=config.get('conf_topk', 20),
            min_steps_before_branch=config.get('min_steps_before_branch', 50),
            branch_cooldown=config.get('branch_cooldown', 100),
            max_branches_per_trace=config.get('max_branches_per_trace', 5)
        )


def compute_branching_distribution(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze branching patterns from completed traces.
    
    Returns statistics about where and when branches occurred.
    """
    branch_points = []
    confidence_at_branches = []
    branch_depths = []
    
    for trace in traces:
        if 'branch_history' in trace:
            for branch in trace['branch_history']:
                branch_points.append(branch['step'])
                confidence_at_branches.append(branch['confidence'])
                branch_depths.append(branch.get('depth', 0))
    
    if not branch_points:
        return {
            'total_branches': 0,
            'avg_branch_step': 0,
            'avg_confidence_at_branch': 0,
            'branch_depth_distribution': {}
        }
    
    return {
        'total_branches': len(branch_points),
        'avg_branch_step': np.mean(branch_points),
        'std_branch_step': np.std(branch_points),
        'avg_confidence_at_branch': np.mean(confidence_at_branches),
        'std_confidence_at_branch': np.std(confidence_at_branches),
        'branch_depth_distribution': dict(zip(*np.unique(branch_depths, return_counts=True))),
        'earliest_branch': min(branch_points),
        'latest_branch': max(branch_points)
    }