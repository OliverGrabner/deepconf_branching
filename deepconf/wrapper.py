"""
DeepThinkLLM implementation with online and offline mode support

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
import numpy as np
from typing import Optional, Dict, Any
import os
import copy

from .outputs import DeepThinkOutput
from .utils import (
    process_batch_results, process_batch_results_offline,
    weighted_majority_vote, compute_all_voting_results, compute_confidence
)
from .branching import BranchingManager, TraceState



class DeepThinkLLM:
    """Enhanced LLM wrapper with deep thinking capabilities"""
    
    def __init__(self, model: str, **vllm_kwargs):
        """
        Initialize DeepThinkLLM
        
        Args:
            model: Model path or name
            **vllm_kwargs: Additional arguments for vLLM initialization
        """
        self.model_name = model
        self.vllm_kwargs = vllm_kwargs
        
        # Initialize vLLM
        default_kwargs = {
            "tensor_parallel_size": len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")),
            "enable_prefix_caching": True,
            "trust_remote_code": True,
        }
        default_kwargs.update(vllm_kwargs)
        
        print("Initializing vLLM engine...")
        llm_init_start = time.time()
        self.llm = LLM(model=model, **default_kwargs)
        llm_init_time = time.time() - llm_init_start
        print(f"vLLM engine initialized in {llm_init_time:.2f} seconds")
        
        # Initialize tokenizer
        print("Initializing tokenizer...")
        tokenizer_init_start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        tokenizer_init_time = time.time() - tokenizer_init_start
        print(f"Tokenizer initialized in {tokenizer_init_time:.2f} seconds")
        
        # Store initialization times
        self.init_times = {
            'llm_init_time': llm_init_time,
            'tokenizer_init_time': tokenizer_init_time
        }
    
    def generate(self, *args, **kwargs):
        """Simple wrapper around vLLM's generate method"""
        return self.llm.generate(*args, **kwargs)
    
    def deepthink(
        self,
        prompt: str,
        mode: str = "offline",
        # Online mode parameters
        warmup_traces: int = 16,
        total_budget: int = 256,
        confidence_percentile: int = 90,
        # Offline mode parameters
        budget: int = 512,
        # Branching mode parameters
        start_traces: int = 8,
        max_traces: int = 32,
        selected_percent: float = 0.60,
        n_iterations: int = 10,
        branch_goal: float = 0.75,
        average_tokens: int = 8000,
        # Common parameters
        window_size: int = 2048,
        sampling_params: Optional[SamplingParams] = None,
        # Multiple voting options
        compute_multiple_voting: bool = True,
        **kwargs
    ) -> DeepThinkOutput:
        """
        Perform deep thinking on a prompt

        Args:
            prompt: Input prompt (prepared string)
            mode: "online" for confidence-based early stopping, "offline" for batch generation, "branching" for dynamic branching
            warmup_traces: Number of warmup traces for online mode
            total_budget: Total budget for online mode
            confidence_percentile: Percentile for confidence threshold in online mode
            budget: Number of traces for offline mode
            start_traces: Initial traces for branching mode
            max_traces: Maximum traces for branching mode
            selected_percent: Top % eligible for branching (branching mode)
            n_iterations: Number of branching check points (branching mode)
            branch_goal: Target completion % for branching (branching mode)
            average_tokens: Historical average tokens (branching mode)
            window_size: Window size for confidence computation
            sampling_params: Custom vLLM sampling parameters
            compute_multiple_voting: Whether to compute multiple voting method results

        Returns:
            DeepThinkOutput containing results
        """
        total_start_time = time.time()
        
        # Create output object
        output = DeepThinkOutput()
        output.mode = mode
        output.llm_init_time = self.init_times['llm_init_time']
        output.tokenizer_init_time = self.init_times['tokenizer_init_time']
        
        # Set configuration
        output.config = {
            "model": self.model_name,
            "mode": mode,
            "window_size": window_size,
            "compute_multiple_voting": compute_multiple_voting,
        }
        
        if mode == "online":
            output.config.update({
                "warmup_traces": warmup_traces,
                "total_budget": total_budget,
                "confidence_percentile": confidence_percentile,
            })
            result = self._deepthink_online(
                prompt, output,
                warmup_traces, total_budget, confidence_percentile,
                window_size, sampling_params
            )
        elif mode == "branching":
            output.config.update({
                "start_traces": start_traces,
                "max_traces": max_traces,
                "selected_percent": selected_percent,
                "n_iterations": n_iterations,
                "branch_goal": branch_goal,
                "average_tokens": average_tokens,
            })
            result = self._deepthink_branching(
                prompt, output,
                start_traces, max_traces, selected_percent,
                n_iterations, branch_goal, average_tokens,
                window_size, sampling_params
            )
        else:
            output.config.update({
                "budget": budget,
            })
            result = self._deepthink_offline(
                prompt, output,
                budget, window_size, sampling_params
            )
        
        # Perform multiple voting analysis if requested
        if compute_multiple_voting and output.all_traces:
            print("Computing multiple voting results...")
            voting_start = time.time()
            output.voting_results = compute_all_voting_results(output.all_traces)
            
            # Set the primary answer to the majority vote result
            if 'majority' in output.voting_results and output.voting_results['majority']:
                output.voted_answer = output.voting_results['majority']['answer']
                output.final_answer = output.voted_answer
            
            voting_time = time.time() - voting_start
            print(f"Multiple voting computed in {voting_time:.2f} seconds")
        
        output.total_time = time.time() - total_start_time
        output.print_summary()
        
        if compute_multiple_voting and output.voting_results:
            output.print_detailed_voting_results()
        
        return output
    
    def _deepthink_online(
        self,
        prompt: str,
        output: DeepThinkOutput,
        warmup_traces: int,
        total_budget: int,
        confidence_percentile: int,
        window_size: int,
        sampling_params: Optional[SamplingParams]
    ) -> DeepThinkOutput:
        """Online deep thinking with confidence-based early stopping"""
        
        processing_start = time.time()
        
        # Warmup phase
        print(f"Starting warmup phase...")
        warmup_gen_start = time.time()
        
        # Generate warmup traces
        warmup_params = copy.deepcopy(sampling_params) 
        warmup_params.n = warmup_traces
        warmup_params.logprobs = 20

        warmup_outputs = self.llm.generate([prompt], warmup_params)
        output.warmup_gen_time = time.time() - warmup_gen_start
        
        # Process warmup results
        warmup_process_start = time.time()
        warmup_result = process_batch_results(warmup_outputs, window_size)
        output.warmup_process_time = time.time() - warmup_process_start
        
        print('Warmup min_confs:', warmup_result['min_confs'])
        output.conf_bar = float(np.percentile(warmup_result['min_confs'],100 - confidence_percentile))
        output.warmup_min_confs = warmup_result['min_confs']
        
        output.warmup_traces = warmup_result['traces']
        output.warmup_tokens = warmup_result['total_tokens']
        
        print(f"Warmup completed: conf_bar={output.conf_bar:.3f}")
        
        # Final phase
        print(f"Starting final phase...")
        final_gen_start = time.time()
        final_params = copy.deepcopy(sampling_params)
        final_params.n = total_budget - warmup_traces
        
        final_outputs = self.llm.generate([prompt], final_params)
        output.final_gen_time = time.time() - final_gen_start
        
        # Process final results
        final_process_start = time.time()
        final_result = process_batch_results(final_outputs, window_size)
        output.final_process_time = time.time() - final_process_start
        
        print('Final min_confs:', final_result['min_confs'])
        output.final_min_confs = final_result['min_confs']
        
        output.final_traces = final_result['traces']
        output.final_tokens = final_result['total_tokens']
        
        # Apply confidence threshold to final traces
        for trace in output.final_traces:
            if trace["min_conf"] < output.conf_bar:
                trace["stop_reason"] = "gconf_threshold"
        
        # Combine all traces
        output.all_traces = output.warmup_traces + output.final_traces
        output.total_tokens = output.warmup_tokens + output.final_tokens
        output.total_traces_count = len(output.all_traces)
        
        # Basic voting (for backward compatibility)
        self._perform_basic_voting(output)
        
        output.processing_time = time.time() - processing_start
        return output
    
    def _deepthink_offline(
        self,
        prompt: str,
        output: DeepThinkOutput,
        budget: int,
        window_size: int,
        sampling_params: Optional[SamplingParams]
    ) -> DeepThinkOutput:
        """Offline deep thinking - generate all traces at once"""
        

        sampling_params.n = budget
        
        # Generate all traces at once
        print(f"Generating {budget} traces...")
        generation_start = time.time()
        vllm_outputs = self.llm.generate([prompt], sampling_params)
        output.generation_time = time.time() - generation_start
        
        # Process results
        processing_start = time.time()
        processed_results = process_batch_results_offline(vllm_outputs, window_size)
        
        output.all_traces = processed_results['traces']
        output.total_tokens = processed_results['total_tokens']
        output.total_traces_count = len(output.all_traces)
        output.avg_tokens_per_trace = output.total_tokens / output.total_traces_count if output.total_traces_count > 0 else 0
        
        # Basic voting (for backward compatibility)
        self._perform_basic_voting(output)
        
        output.processing_time = time.time() - processing_start
        return output
    
    def _perform_basic_voting(self, output: DeepThinkOutput):
        """Perform basic weighted majority voting (for backward compatibility)"""
        voting_answers = []
        voting_weights = []
        
        if output.mode == "online":
            # Add warmup traces above threshold
            for trace in output.warmup_traces:
                if trace.get('min_conf', 0) >= output.conf_bar and trace.get('extracted_answer'):
                    voting_answers.append(trace['extracted_answer'])
                    voting_weights.append(trace.get('min_conf', 1.0))
            
            # Add final traces (skip early stopped ones)
            for trace in output.final_traces:
                if trace.get('stop_reason') == 'gconf_threshold':
                    continue
                if trace.get('extracted_answer'):
                    voting_answers.append(trace['extracted_answer'])
                    voting_weights.append(trace.get('min_conf', 1.0))
        else:
            # Offline mode - use all traces with valid answers
            for trace in output.all_traces:
                if trace.get('extracted_answer'):
                    voting_answers.append(trace['extracted_answer'])
                    voting_weights.append(1.0)
        
        output.voting_answers = voting_answers
        output.voting_weights = voting_weights
        
        # Get voted answer (basic method)
        output.voted_answer = weighted_majority_vote(voting_answers, voting_weights)
        output.final_answer = output.voted_answer
        
        # Calculate token statistics
        if output.mode == "online":
            output.avg_tokens_per_warmup_trace = output.warmup_tokens / len(output.warmup_traces) if output.warmup_traces else 0
            output.avg_tokens_per_final_trace = output.final_tokens / len(output.final_traces) if output.final_traces else 0
        
        print(f'Basic voting candidates: {len(voting_answers)}')
        if voting_answers:
            print(f'Sample voting answers: {voting_answers[:5]}')

    def _deepthink_branching(
        self,
        prompt: str,
        output: DeepThinkOutput,
        start_traces: int,
        max_traces: int,
        selected_percent: float,
        n_iterations: int,
        branch_goal: float,
        average_tokens: int,
        window_size: int,
        sampling_params: Optional[SamplingParams]
    ) -> DeepThinkOutput:
        """Branching self-consistency - dynamic trace branching during generation"""

        processing_start = time.time()

        # Initialize branching manager
        manager = BranchingManager(
            start_traces=start_traces,
            max_traces=max_traces,
            selected_percent=selected_percent,
            n_iterations=n_iterations,
            branch_goal=branch_goal,
            average_tokens=average_tokens,
            tail_window=window_size
        )

        # Store branching config in output
        output.branching_config = {
            'start_traces': start_traces,
            'max_traces': max_traces,
            'selected_percent': selected_percent,
            'n_iterations': n_iterations,
            'branch_goal': branch_goal,
            'average_tokens': average_tokens,
            'stride': manager.stride,
            'branch_deadline_tokens': manager.branch_deadline_tokens
        }

        # Initialize starting traces
        manager.initialize_traces(start_traces)

        # Prepare initial prompts (all identical at start)
        active_prompts = [prompt] * start_traces

        print(f"\nStarting branching generation with {start_traces} traces...")
        generation_start = time.time()

        # Generation loop - iterate until branching deadline
        for iteration in range(n_iterations):
            manager.print_status()

            # Generate next chunk (stride tokens) for all active traces
            chunk_params = copy.deepcopy(sampling_params)
            chunk_params.n = 1  # Generate 1 completion per prompt
            chunk_params.max_tokens = manager.stride

            print(f"  Generating {manager.stride} tokens for {len(active_prompts)} traces...")
            chunk_start = time.time()

            # Generate for all prompts in batch (optimized for parallel processing)
            batch_results = self.llm.generate(active_prompts, chunk_params)

            chunk_time = time.time() - chunk_start
            print(f"  Chunk generation took {chunk_time:.2f}s")

            # Update trace states with new generations
            for i, (trace, result) in enumerate(zip(manager.active_traces, batch_results)):
                if len(result.outputs) > 0:
                    gen_output = result.outputs[0]

                    # Append new text
                    new_text = gen_output.text
                    trace.current_text += new_text

                    # Append new token IDs
                    if gen_output.token_ids:
                        trace.current_token_ids.extend(gen_output.token_ids)

                    # Compute and append confidences
                    if gen_output.logprobs:
                        new_confs = compute_confidence(gen_output.logprobs)
                        trace.current_confs.extend(new_confs)

                    # Update prompt for next iteration
                    active_prompts[i] = prompt + trace.current_text

                    # Check if complete
                    if gen_output.finish_reason and gen_output.finish_reason != 'length':
                        trace.is_complete = True

            # Decide whether to branch
            if manager.should_branch():
                print(f"  Evaluating branching candidates...")

                # Select candidates based on tail confidence
                candidates = manager.select_branch_candidates()
                print(f"    Top {selected_percent*100:.0f}% candidates: {len(candidates)}")

                # Select which traces to branch
                branches_to_create = manager.select_branches_to_create(candidates)

                if branches_to_create:
                    print(f"    Creating {len(branches_to_create)} new branches...")

                    # Create new branches
                    new_traces = manager.create_branches(
                        parent_indices=branches_to_create,
                        timestamp=time.time()
                    )

                    # Add corresponding prompts for new branches
                    trace_map = {t.trace_idx: t for t in manager.active_traces}
                    for new_trace in new_traces:
                        # New branch starts with parent's current text
                        branch_prompt = prompt + new_trace.current_text
                        active_prompts.append(branch_prompt)

                    print(f"    Total active traces: {len(manager.active_traces)}")
            else:
                if len(manager.active_traces) >= max_traces:
                    print(f"  Max traces reached ({max_traces}), no more branching")
                else:
                    print(f"  Past branching deadline, no more branching")

            # Advance to next iteration
            manager.advance_iteration()

        # Continue generation until completion (no more branching)
        print(f"\nBranching phase complete. Continuing generation to completion...")
        print(f"  {len(active_prompts)} traces generating...")

        # Generate until max_tokens for all remaining traces
        final_params = copy.deepcopy(sampling_params)
        final_params.n = 1
        final_params.max_tokens = sampling_params.max_tokens - manager.stride * n_iterations

        # Batch generate for all traces (optimized for parallel processing)
        final_results = self.llm.generate(active_prompts, final_params)

        # Update traces with final generation
        from .utils import extract_answer

        for i, (trace, result) in enumerate(zip(manager.active_traces, final_results)):
            if len(result.outputs) > 0:
                gen_output = result.outputs[0]

                # Append final text
                trace.current_text += gen_output.text

                # Append final token IDs
                if gen_output.token_ids:
                    trace.current_token_ids.extend(gen_output.token_ids)

                # Append final confidences
                if gen_output.logprobs:
                    new_confs = compute_confidence(gen_output.logprobs)
                    trace.current_confs.extend(new_confs)

                trace.is_complete = True

        output.generation_time = time.time() - generation_start

        # Convert TraceState objects to trace dictionaries for output
        all_traces = []
        total_tokens = 0
        total_tokens_generated = 0  # NEW: Only count newly generated tokens

        for trace in manager.active_traces:
            extracted_answer = extract_answer(trace.current_text)

            # Calculate tokens generated by THIS trace (not inherited from parent)
            tokens_generated = len(trace.current_token_ids) - trace.generation_started_at_tokens

            trace_dict = {
                'trace_idx': trace.trace_idx,
                'parent_idx': trace.parent_idx,
                'text': trace.current_text,
                'token_ids': trace.current_token_ids,
                'num_tokens': len(trace.current_token_ids),  # Total including inherited
                'tokens_generated': tokens_generated,  # NEW: Only new tokens
                'confs': trace.current_confs,
                'extracted_answer': extracted_answer,
                'is_complete': trace.is_complete,
                'generation_started_at_iteration': trace.generation_started_at_iteration,
                'generation_started_at_tokens': trace.generation_started_at_tokens
            }

            all_traces.append(trace_dict)
            total_tokens += trace_dict['num_tokens']  # Keep old for compatibility
            total_tokens_generated += tokens_generated  # NEW: Accurate count

        output.all_traces = all_traces
        output.total_tokens = total_tokens  # Includes inherited tokens (for compatibility)
        output.total_tokens_generated = total_tokens_generated  # NEW: Only new tokens
        output.total_traces_count = len(all_traces)
        output.avg_tokens_per_trace = total_tokens / len(all_traces) if all_traces else 0
        output.avg_tokens_generated_per_trace = total_tokens_generated / len(all_traces) if all_traces else 0

        # Get genealogy information
        output.branch_genealogy = manager.get_genealogy()
        output.branch_events = [
            {
                'iteration': e.iteration,
                'parent_trace_idx': e.parent_trace_idx,
                'child_trace_idx': e.child_trace_idx,
                'branch_point_tokens': e.branch_point_tokens,
                'parent_tail_confidence': e.parent_tail_confidence
            }
            for e in manager.branch_events
        ]

        # Perform basic voting
        self._perform_basic_voting(output)

        output.processing_time = time.time() - processing_start

        print(f"\nBranching generation complete:")
        print(f"  Total traces: {len(all_traces)}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Branch events: {len(manager.branch_events)}")

        return output