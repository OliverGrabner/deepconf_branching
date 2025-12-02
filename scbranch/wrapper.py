"""
SCLLM - Self-Consistency LLM wrapper with branching support

Supports three modes:
- offline: Traditional self-consistency (generate N independent traces)
- branching: Dynamic branching during generation
- peak_branching: Branch from confidence peaks
"""
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
import numpy as np
from typing import Optional, Dict, Any
import os
import copy

from .outputs import SCOutput
from .utils import (
    process_batch_results_offline, simple_majority_vote, compute_confidence,
    extract_answer, prepare_prompt_from_tokens
)
from .branching import BranchingManager, TraceState
from .peak_branching import PeakBranchingManager, PeakTrace



class SCLLM:
    """Self-Consistency LLM wrapper with branching support"""
    
    def __init__(self, model: str, **vllm_kwargs):
        """
        Initialize SCLLM
        
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
        # Offline mode (traditional SC) parameters
        budget: int = 512,
        # Branching mode parameters
        start_traces: int = 8,
        max_traces: int = 32,
        selected_percent: float = 0.60,
        n_iterations: int = 10,
        branch_goal: float = 0.75,
        average_tokens: int = 8000,
        # Peak branching mode parameters
        initial_traces: int = 8,
        peak_max_traces: int = 32,
        confidence_threshold: float = 1.5,
        peak_window_size: int = 512,
        min_peak_distance: int = 256,
        peak_selection_ratio: float = 0.8,
        exclusion_zone_size: int = 200,
        # Common parameters
        window_size: int = 2048,
        sampling_params: Optional[SamplingParams] = None,
        **kwargs
    ) -> SCOutput:
        """
        Run self-consistency with optional branching

        Args:
            prompt: Input prompt (prepared string)
            mode: "offline" for traditional SC, "branching" for dynamic branching,
                  "peak_branching" for confidence peak branching
            budget: Number of traces for offline mode
            start_traces: Initial traces for branching mode
            max_traces: Maximum traces for branching mode
            selected_percent: Top % eligible for branching (branching mode)
            n_iterations: Number of branching check points (branching mode)
            branch_goal: Target completion % for branching (branching mode)
            average_tokens: Historical average tokens (branching mode)
            initial_traces: Initial traces for peak branching mode
            peak_max_traces: Maximum total traces (stops when next doubling would exceed)
            confidence_threshold: Minimum confidence for peaks (peak branching mode)
            peak_window_size: Window size for peak detection (peak branching mode)
            min_peak_distance: Minimum distance between peaks (peak branching mode)
            peak_selection_ratio: Valid range for peaks (peak branching mode)
            exclusion_zone_size: Size of exclusion zone around used peaks (peak branching mode)
            window_size: Window size for confidence computation
            sampling_params: Custom vLLM sampling parameters

        Returns:
            SCOutput containing results
        """
        total_start_time = time.time()

        # Create output object
        output = SCOutput()
        output.mode = mode
        output.llm_init_time = self.init_times['llm_init_time']
        output.tokenizer_init_time = self.init_times['tokenizer_init_time']

        # Set configuration
        output.config = {
            "model": self.model_name,
            "mode": mode,
            "window_size": window_size,
        }
        
        if mode == "branching":
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
        elif mode == "peak_branching":
            output.config.update({
                "initial_traces": initial_traces,
                "max_traces": peak_max_traces,
                "confidence_threshold": confidence_threshold,
                "peak_window_size": peak_window_size,
                "min_peak_distance": min_peak_distance,
                "peak_selection_ratio": peak_selection_ratio,
                "exclusion_zone_size": exclusion_zone_size,
            })
            result = self._deepthink_peak_branching(
                prompt, output,
                initial_traces, peak_max_traces, confidence_threshold,
                peak_window_size, min_peak_distance, peak_selection_ratio,
                exclusion_zone_size, window_size, sampling_params
            )
        else:
            output.config.update({
                "budget": budget,
            })
            result = self._deepthink_offline(
                prompt, output,
                budget, window_size, sampling_params
            )

        output.total_time = time.time() - total_start_time
        output.print_summary()

        return output

    def _deepthink_offline(
        self,
        prompt: str,
        output: SCOutput,
        budget: int,
        window_size: int,
        sampling_params: Optional[SamplingParams]
    ) -> SCOutput:
        """Offline deep thinking - generate all traces at once"""

        import copy
        # Create a copy to avoid modifying the original
        offline_params = copy.deepcopy(sampling_params)
        offline_params.n = budget

        # Cap max_tokens at reasonable limit for traditional SC
        offline_params.max_tokens = min(8000, offline_params.max_tokens)
        offline_params.stop = ["}\n\n", "}\n"]  # Stop after completing \boxed{answer}

        # Generate all traces at once
        print(f"Generating {budget} traces...")
        print(f"  Max tokens: {offline_params.max_tokens}")
        print(f"  Stop sequences: {offline_params.stop}")
        generation_start = time.time()
        vllm_outputs = self.llm.generate([prompt], offline_params)
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
    
    def _perform_basic_voting(self, output: SCOutput):
        """Perform simple majority voting"""
        voting_answers = []

        # Collect all valid answers
        for trace in output.all_traces:
            if trace.get('extracted_answer'):
                voting_answers.append(trace['extracted_answer'])

        output.voting_answers = voting_answers

        # Simple majority vote
        output.voted_answer = simple_majority_vote(voting_answers)
        output.final_answer = output.voted_answer

        print(f'Voting candidates: {len(voting_answers)}')
        if voting_answers:
            print(f'Sample answers: {voting_answers[:5]}')

    def _deepthink_branching(
        self,
        prompt: str,
        output: SCOutput,
        start_traces: int,
        max_traces: int,
        selected_percent: float,
        n_iterations: int,
        branch_goal: float,
        average_tokens: int,
        window_size: int,
        sampling_params: Optional[SamplingParams]
    ) -> SCOutput:
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
        # Cap final generation at reasonable limit (same as traditional SC)
        remaining_budget = sampling_params.max_tokens - manager.stride * n_iterations
        final_params.max_tokens = min(8000, remaining_budget)
        final_params.stop = ["}\n\n", "}\n"]  # Stop after completing \boxed{answer}

        print(f"  Final generation max_tokens: {final_params.max_tokens}")
        print(f"  Stop sequences: {final_params.stop}")

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

    def _deepthink_peak_branching(
        self,
        prompt: str,
        output: SCOutput,
        initial_traces: int,
        peak_max_traces: int,
        confidence_threshold: float,
        peak_window_size: int,
        min_peak_distance: int,
        peak_selection_ratio: float,
        exclusion_zone_size: int,
        window_size: int,
        sampling_params: Optional[SamplingParams]
    ) -> SCOutput:
        """Multi-stage peak branching - branch from confidence peaks with doubling strategy"""

        processing_start = time.time()

        # Initialize peak branching manager with multi-stage support
        manager = PeakBranchingManager(
            initial_traces=initial_traces,
            max_traces=peak_max_traces,
            confidence_threshold=confidence_threshold,
            window_size=peak_window_size,
            min_peak_distance=min_peak_distance,
            peak_selection_ratio=peak_selection_ratio,
            exclusion_zone_size=exclusion_zone_size
        )

        # Store config in output
        output.peak_branching_config = {
            'initial_traces': initial_traces,
            'max_traces': peak_max_traces,
            'confidence_threshold': confidence_threshold,
            'peak_window_size': peak_window_size,
            'min_peak_distance': min_peak_distance,
            'peak_selection_ratio': peak_selection_ratio,
            'exclusion_zone_size': exclusion_zone_size,
            'branching_stages': manager.branching_stages
        }

        # Phase 1: Generate initial traces
        print(f"\n=== Phase 1: Generating {initial_traces} initial traces ===")
        generation_start = time.time()

        # Generate all initial traces at once
        initial_params = copy.deepcopy(sampling_params)
        initial_params.n = initial_traces
        initial_params.logprobs = 20  # Need logprobs for confidence
        initial_params.max_tokens = 8000  # Reasonable limit for initial traces
        initial_params.stop = ["}\n\n", "}\n"]  # Stop after completing \boxed{answer}

        print(f"Generating initial traces...")
        print(f"  Max tokens: {initial_params.max_tokens}")
        print(f"  Stop sequences: {initial_params.stop}")
        print(f"  Prompt length: {len(prompt)} chars")

        vllm_outputs = self.llm.generate([prompt], initial_params)
        initial_gen_time = time.time() - generation_start

        # Process initial traces
        print("Processing initial traces...")
        for i, vllm_output in enumerate(vllm_outputs[0].outputs):
            text = vllm_output.text
            token_ids = vllm_output.token_ids
            logprobs = vllm_output.logprobs

            print(f"\nInitial trace {i}:")
            print(f"  Generated {len(token_ids)} tokens")
            print(f"  Finish reason: {vllm_output.finish_reason}")

            # Calculate confidence scores
            confs = compute_confidence(logprobs) if logprobs else []

            # Extract answer
            extracted_answer = extract_answer(text)
            if extracted_answer:
                print(f"  Found answer: {extracted_answer}")

            # Create initial trace
            trace = manager.create_initial_trace(
                text=text,
                token_ids=token_ids,
                confs=confs,
                extracted_answer=extracted_answer
            )

            print(f"  Trace {i}: {len(token_ids)} tokens, answer: {extracted_answer}")

        # Phase 2: Multi-Stage Branching
        print(f"\n=== Phase 2: Multi-Stage Branching ===")
        total_branch_gen_time = 0

        # Execute each branching stage
        for stage_idx, num_branches in enumerate(manager.branching_stages):
            stage_num = stage_idx + 1

            # Run branching stage to find peaks
            selected_peaks = manager.run_branching_stage(stage_idx, num_branches)

            if not selected_peaks:
                print(f"Stage {stage_num}: No valid peaks found, skipping...")
                continue

            # Generate branches for this stage
            print(f"\nGenerating {len(selected_peaks)} branches for stage {stage_num}...")
            stage_gen_start = time.time()

            # Prepare branch prompts
            branch_prompts_info = manager.prepare_branch_prompts(selected_peaks)

            # Generate branches with prefix caching
            for i, branch_info in enumerate(branch_prompts_info):
                print(f"\nStage {stage_num} Branch {i+1}/{len(branch_prompts_info)}:")
                print(f"  From trace {branch_info['parent_trace_idx']} (stage {branch_info['parent_stage']}) @ position {branch_info['branch_point']}")
                print(f"  Parent confidence: {branch_info['parent_confidence']:.3f}")

                # Reconstruct prompt from prefix tokens
                prefix_prompt = prepare_prompt_from_tokens(
                    branch_info['prompt_tokens'],
                    self.tokenizer
                )

                # Log branch details
                print(f"  Prefix tokens: {len(branch_info['prompt_tokens'])} tokens")
                print(f"  Prefix prompt length: {len(prefix_prompt)} chars")
                print(f"  Remaining token budget: {64000 - len(branch_info['prompt_tokens'])} tokens")

                # Show last 200 chars of prefix to see where we're branching from
                print(f"  Branching from: ...{prefix_prompt[-200:]}" if len(prefix_prompt) > 200 else f"  Branching from: {prefix_prompt}")

                # Generate continuation from branch point
                branch_params = copy.deepcopy(sampling_params)
                branch_params.n = 1
                branch_params.logprobs = 20
                # Cap at 4000 tokens to prevent runaway generation
                # Use min() not max() - we want the SMALLER of 4000 or remaining budget
                branch_params.max_tokens = min(4000, 64000 - len(branch_info['prompt_tokens']))
                branch_params.stop = ["}\n\n", "}\n"]  # Stop after completing \boxed{answer}

                print(f"  Branch max_tokens: {branch_params.max_tokens}")
                print(f"  Generating branch continuation...")

                branch_output = self.llm.generate([prefix_prompt], branch_params)
                branch_vllm = branch_output[0].outputs[0]

                # Process branch output
                branch_text = branch_vllm.text
                branch_token_ids = branch_vllm.token_ids
                branch_logprobs = branch_vllm.logprobs

                # Calculate confidence for full branch
                branch_confs = compute_confidence(branch_logprobs) if branch_logprobs else []

                # Extract answer from branch
                branch_answer = extract_answer(branch_text)

                # Log branch results
                print(f"  Branch generated {len(branch_token_ids)} tokens")
                print(f"  Finish reason: {branch_vllm.finish_reason}")
                if branch_answer:
                    print(f"  Found answer: {branch_answer}")

                # Create branch trace with stage information
                branch_trace = manager.create_branch_trace(
                    parent_trace_idx=branch_info['parent_trace_idx'],
                    branch_point=branch_info['branch_point'],
                    stage=stage_num,
                    new_text=branch_text,
                    new_token_ids=branch_token_ids,
                    new_confs=branch_confs,
                    extracted_answer=branch_answer
                )

                print(f"  Generated {branch_trace.tokens_generated} new tokens")
                print(f"  Branch answer: {branch_answer}")

            stage_gen_time = time.time() - stage_gen_start
            total_branch_gen_time += stage_gen_time
            print(f"\nStage {stage_num} generation completed in {stage_gen_time:.2f}s")

        # Compile results
        output.generation_time = initial_gen_time + total_branch_gen_time

        # Convert traces to output format
        all_traces = []
        total_tokens_generated = 0

        for trace in manager.traces:
            trace_dict = {
                'trace_idx': trace.trace_idx,
                'stage': trace.stage,  # Multi-stage support
                'depth': trace.depth,  # Same as stage for compatibility
                'parent_idx': trace.parent_idx,
                'branch_point_tokens': trace.branch_point_tokens,
                'text': trace.text,
                'token_ids': trace.token_ids,
                'num_tokens': trace.total_tokens,
                'tokens_generated': trace.tokens_generated,
                'confs': trace.confs,
                'extracted_answer': trace.extracted_answer,
                'confidence_peaks': [
                    {
                        'position': p.position,
                        'confidence': p.confidence
                    }
                    for p in trace.confidence_peaks
                ]
            }
            all_traces.append(trace_dict)
            total_tokens_generated += trace.tokens_generated

        output.all_traces = all_traces
        output.total_tokens = total_tokens_generated  # Only count generated tokens
        output.total_traces_count = len(all_traces)
        output.avg_tokens_per_trace = total_tokens_generated / len(all_traces) if all_traces else 0

        # Store peak branching statistics
        output.peak_branching_stats = manager.get_statistics()

        # Simple majority voting
        voting_answers = [trace['extracted_answer'] for trace in all_traces if trace['extracted_answer']]
        output.voting_answers = voting_answers
        output.voted_answer = simple_majority_vote(voting_answers)
        output.final_answer = output.voted_answer

        # Print summary
        manager.print_summary()

        output.processing_time = time.time() - processing_start

        return output