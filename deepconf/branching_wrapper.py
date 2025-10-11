"""
DeepThinkLLM with confidence-based branching support

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import os
import copy
from queue import Queue
from dataclasses import dataclass
import threading

from .outputs import DeepThinkOutput
from .utils import (
    process_batch_results_offline, 
    weighted_majority_vote, compute_all_voting_results,
    extract_answer
)
from .branching_processors import WrappedBranchingProcessor, compute_branching_distribution


@dataclass
class BranchingTrace:
    """Represents a trace that may spawn branches"""
    trace_id: str
    parent_id: Optional[str]
    prompt: str
    prefix_tokens: List[int]
    depth: int
    confidence_history: List[float]
    branch_history: List[Dict[str, Any]]
    

class BranchingDeepThinkLLM:
    """Enhanced DeepThinkLLM with confidence-based branching capabilities"""
    
    def __init__(self, model: str, **vllm_kwargs):
        """
        Initialize BranchingDeepThinkLLM
        
        Args:
            model: Model path or name
            **vllm_kwargs: Additional arguments for vLLM initialization
        """
        self.model_name = model
        self.vllm_kwargs = vllm_kwargs
        
        # Initialize vLLM with branching processor
        default_kwargs = {
            "tensor_parallel_size": len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")),
            "enable_prefix_caching": True,
            "trust_remote_code": True,
        }
        default_kwargs.update(vllm_kwargs)
        
        print("Initializing vLLM engine with branching support...")
        llm_init_start = time.time()
        self.llm = LLM(
            model=model, 
            logits_processors=[WrappedBranchingProcessor], 
            **default_kwargs
        )
        llm_init_time = time.time() - llm_init_start
        print(f"vLLM engine initialized in {llm_init_time:.2f} seconds")
        
        # Initialize tokenizer
        print("Initializing tokenizer...")
        tokenizer_init_start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        tokenizer_init_time = time.time() - tokenizer_init_start
        print(f"Tokenizer initialized in {tokenizer_init_time:.2f} seconds")
        
        self.init_times = {
            'llm_init_time': llm_init_time,
            'tokenizer_init_time': tokenizer_init_time
        }
    
    def generate(self, *args, **kwargs):
        """Simple wrapper around vLLM's generate method"""
        return self.llm.generate(*args, **kwargs)
    
    def branching_deepthink(
        self,
        prompt: str,
        initial_branches: int = 4,
        max_total_branches: int = 64,
        max_depth: int = 3,
        # Branching parameters
        base_branch_prob: float = 0.1,
        max_branch_prob: float = 0.9,
        confidence_threshold: float = 2.0,
        min_steps_before_branch: int = 50,
        branch_cooldown: int = 100,
        max_branches_per_trace: int = 5,
        # Generation parameters
        window_size: int = 128,
        sampling_params: Optional[SamplingParams] = None,
        **kwargs
    ) -> DeepThinkOutput:
        """
        Perform deep thinking with confidence-based branching.
        
        Instead of generating a fixed number of traces, this dynamically spawns
        new traces from high-confidence states.
        
        Args:
            prompt: Input prompt
            initial_branches: Number of initial traces to start with
            max_total_branches: Maximum total traces to generate
            max_depth: Maximum branching depth
            base_branch_prob: Minimum branching probability
            max_branch_prob: Maximum branching probability at high confidence
            confidence_threshold: Confidence level for max branching probability
            min_steps_before_branch: Minimum tokens before first branch
            branch_cooldown: Minimum tokens between branches
            max_branches_per_trace: Max branches per individual trace
            window_size: Sliding window for confidence computation
            sampling_params: vLLM sampling parameters
            
        Returns:
            DeepThinkOutput with branching analysis
        """
        total_start_time = time.time()
        
        # Create output object
        output = DeepThinkOutput()
        output.mode = "branching"
        output.llm_init_time = self.init_times['llm_init_time']
        output.tokenizer_init_time = self.init_times['tokenizer_init_time']
        
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=32000,
                logprobs=20,
            )
        
        # Set configuration
        output.config = {
            "model": self.model_name,
            "mode": "branching",
            "initial_branches": initial_branches,
            "max_total_branches": max_total_branches,
            "max_depth": max_depth,
            "branching_params": {
                "base_branch_prob": base_branch_prob,
                "max_branch_prob": max_branch_prob,
                "confidence_threshold": confidence_threshold,
                "min_steps_before_branch": min_steps_before_branch,
                "branch_cooldown": branch_cooldown,
                "max_branches_per_trace": max_branches_per_trace,
            }
        }
        
        # Since vLLM doesn't support dynamic spawning mid-generation,
        # we'll simulate it by doing multiple rounds of generation
        result = self._simulate_branching(
            prompt, output,
            initial_branches, max_total_branches, max_depth,
            base_branch_prob, max_branch_prob, confidence_threshold,
            min_steps_before_branch, branch_cooldown, max_branches_per_trace,
            window_size, sampling_params
        )
        
        # Compute voting results
        if output.all_traces:
            print("Computing voting results...")
            voting_start = time.time()
            output.voting_results = compute_all_voting_results(output.all_traces)
            
            if 'majority' in output.voting_results and output.voting_results['majority']:
                output.voted_answer = output.voting_results['majority']['answer']
                output.final_answer = output.voted_answer
            
            voting_time = time.time() - voting_start
            print(f"Voting computed in {voting_time:.2f} seconds")
        
        # Compute branching statistics
        output.branching_stats = compute_branching_distribution(output.all_traces)
        
        output.total_time = time.time() - total_start_time
        output.print_summary()
        
        if output.voting_results:
            output.print_detailed_voting_results()
        
        # Print branching statistics
        print("\n=== Branching Statistics ===")
        for key, value in output.branching_stats.items():
            print(f"{key}: {value}")
        
        return output
    
    def _simulate_branching(
        self,
        prompt: str,
        output: DeepThinkOutput,
        initial_branches: int,
        max_total_branches: int,
        max_depth: int,
        base_branch_prob: float,
        max_branch_prob: float,
        confidence_threshold: float,
        min_steps_before_branch: int,
        branch_cooldown: int,
        max_branches_per_trace: int,
        window_size: int,
        sampling_params: SamplingParams
    ) -> DeepThinkOutput:
        """
        Simulate branching by analyzing partial generations and spawning new ones.
        
        This is a simplified version that generates traces in rounds.
        """
        processing_start = time.time()
        
        all_traces = []
        total_tokens = 0
        trace_counter = 0
        
        # Round 1: Generate initial traces
        print(f"Generating {initial_branches} initial traces...")
        gen_start = time.time()
        
        initial_params_list = []
        base_seed = time.time_ns()
        
        for i in range(initial_branches):
            params = copy.deepcopy(sampling_params)
            params.logprobs = 20
            params.seed = base_seed + i
            # Add branching configuration
            params.extra_args = {
                "branching_config": {
                    "base_branch_prob": base_branch_prob,
                    "max_branch_prob": max_branch_prob,
                    "confidence_threshold": confidence_threshold,
                    "conf_group_size": window_size,
                    "conf_topk": 20,
                    "min_steps_before_branch": min_steps_before_branch,
                    "branch_cooldown": branch_cooldown,
                    "max_branches_per_trace": max_branches_per_trace,
                }
            }
            initial_params_list.append(params)
        
        initial_outputs = self.llm.generate(
            [prompt for _ in range(initial_branches)], 
            initial_params_list
        )
        
        output.generation_time = time.time() - gen_start
        
        # Process initial results
        processed_results = process_batch_results_offline(initial_outputs, window_size)
        
        for i, trace in enumerate(processed_results['traces']):
            trace['trace_id'] = f"trace_{trace_counter}"
            trace['parent_id'] = None
            trace['depth'] = 0
            trace['branch_history'] = []
            all_traces.append(trace)
            total_tokens += trace['num_tokens']
            trace_counter += 1
        
        # Analyze traces for potential branching points
        print("\nAnalyzing traces for high-confidence regions...")
        branch_candidates = self._analyze_branching_opportunities(
            all_traces, confidence_threshold, window_size
        )
        
        # Round 2: Generate branches from high-confidence points
        if branch_candidates and trace_counter < max_total_branches:
            num_branches = min(len(branch_candidates), max_total_branches - trace_counter)
            print(f"\nIdentified {len(branch_candidates)} branching opportunities")
            print(f"Generating {num_branches} branch traces...")
            
            # For simplicity, we'll just generate new full traces with different seeds
            # In a real implementation, we'd use prefix caching to continue from branch points
            branch_params_list = []
            
            for i, candidate in enumerate(branch_candidates[:num_branches]):
                params = copy.deepcopy(sampling_params)
                params.logprobs = 20
                # Use a seed influenced by the parent trace and branch point
                params.seed = base_seed + 1000 + i + candidate['confidence_peak'] * 1000
                branch_params_list.append(params)
            
            branch_outputs = self.llm.generate(
                [prompt for _ in range(num_branches)], 
                branch_params_list
            )
            
            # Process branch results
            branch_results = process_batch_results_offline(branch_outputs, window_size)
            
            for i, trace in enumerate(branch_results['traces'][:num_branches]):
                trace['trace_id'] = f"trace_{trace_counter}"
                trace['parent_id'] = branch_candidates[i]['parent_trace_id']
                trace['depth'] = 1
                trace['branch_history'] = [{
                    'step': branch_candidates[i]['peak_position'],
                    'confidence': branch_candidates[i]['confidence_peak'],
                    'parent_trace': branch_candidates[i]['parent_trace_id']
                }]
                all_traces.append(trace)
                total_tokens += trace['num_tokens']
                trace_counter += 1
        
        # Store results
        output.all_traces = all_traces
        output.total_tokens = total_tokens
        output.total_traces_count = len(all_traces)
        output.avg_tokens_per_trace = total_tokens / len(all_traces) if all_traces else 0
        
        # Basic voting
        self._perform_basic_voting(output)
        
        output.processing_time = time.time() - processing_start
        return output
    
    def _analyze_branching_opportunities(
        self, 
        traces: List[Dict[str, Any]], 
        confidence_threshold: float,
        window_size: int
    ) -> List[Dict[str, Any]]:
        """
        Analyze traces to find high-confidence regions suitable for branching.
        
        Returns list of branching candidates sorted by confidence.
        """
        candidates = []
        
        for trace in traces:
            if 'confs' not in trace or not trace['confs']:
                continue
            
            confs = trace['confs']
            if len(confs) < window_size:
                continue
            
            # Find peaks in sliding window confidence
            for i in range(window_size, len(confs) - window_size):
                window = confs[i-window_size//2:i+window_size//2]
                avg_conf = np.mean(window)
                
                # Check if this is a local maximum
                if avg_conf > confidence_threshold:
                    # Simple peak detection
                    if i > 0 and i < len(confs) - 1:
                        prev_window = confs[max(0, i-window_size):i]
                        next_window = confs[i:min(len(confs), i+window_size)]
                        
                        if np.mean(prev_window) < avg_conf and np.mean(next_window) < avg_conf:
                            candidates.append({
                                'parent_trace_id': trace['trace_id'],
                                'peak_position': i,
                                'confidence_peak': avg_conf,
                                'text_up_to_peak': trace['text'][:i] if 'text' in trace else '',
                            })
        
        # Sort by confidence and return top candidates
        candidates.sort(key=lambda x: x['confidence_peak'], reverse=True)
        return candidates
    
    def _perform_basic_voting(self, output: DeepThinkOutput):
        """Perform basic voting on all traces"""
        voting_answers = []
        voting_weights = []
        
        for trace in output.all_traces:
            if trace.get('extracted_answer'):
                voting_answers.append(trace['extracted_answer'])
                # Weight by depth (prefer deeper traces) and confidence
                depth_weight = 1.0 + trace.get('depth', 0) * 0.1
                voting_weights.append(depth_weight)
        
        output.voting_answers = voting_answers
        output.voting_weights = voting_weights
        
        if voting_answers:
            output.voted_answer = weighted_majority_vote(voting_answers, voting_weights)
            output.final_answer = output.voted_answer
        
        print(f'Voting candidates: {len(voting_answers)}')
        if voting_answers:
            print(f'Sample answers: {voting_answers[:5]}')