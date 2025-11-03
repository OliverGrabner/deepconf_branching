# Project Organization

This document explains the clean, organized structure of the DeepConf codebase.

## Directory Structure

```
deepconf_branching/
├── README.md                      # Main project documentation
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
├── test.py                        # Test script (legacy)
│
├── deepconf/                      # Core package
│   ├── __init__.py               # Public API exports
│   ├── wrapper.py                # DeepThinkLLM class
│   ├── utils.py                  # Utilities and voting functions
│   └── outputs.py                # Output dataclasses
│
├── scripts/                       # Executable scripts
│   ├── run_traditional_sc_aime25.py      # Main SC experiment
│   ├── test_sc_single_question.py        # Quick test script
│   ├── analyze_sc_results.py             # Result analysis
│   ├── visualize_sc_results.py           # ASCII visualization
│   ├── visualize_trace_confidence.py     # Confidence graphs
│   └── run_sc_experiments.sh             # Automated experiments
│
├── examples/                      # Usage examples
│   ├── example_offline.py        # Offline mode example
│   └── example_online.py         # Online mode example
│
└── docs/                          # Documentation
    ├── QUICKSTART.md             # Quick reference
    ├── SELF_CONSISTENCY.md       # SC implementation guide
    ├── TRACE_VISUALIZATION.md    # Visualization guide
    ├── IMPLEMENTATION.md         # Technical details
    ├── CHANGELOG.md              # Feature changes
    └── PROJECT_ORGANIZATION.md   # This file
```

## File Purposes

### Root Level

| File | Purpose |
|------|---------|
| `README.md` | Main documentation with quickstart and navigation |
| `requirements.txt` | All Python dependencies |
| `setup.py` | Package installation configuration |
| `test.py` | Legacy test script (from original repo) |

### `deepconf/` - Core Package

| File | Purpose | Key Contents |
|------|---------|--------------|
| `__init__.py` | Public API | Exports: `DeepThinkLLM`, `prepare_prompt`, `equal_func` |
| `wrapper.py` | Main LLM wrapper | `DeepThinkLLM` class, online/offline modes |
| `utils.py` | Utility functions | 7 voting methods, prompt preparation, evaluation |
| `outputs.py` | Output structures | `DeepThinkOutput` dataclass |

### `scripts/` - Executable Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_traditional_sc_aime25.py` | Main SC experiment | Run full AIME25 evaluation |
| `test_sc_single_question.py` | Quick test | Test on 1 question (8 traces) |
| `analyze_sc_results.py` | Result analysis | Deep dive into patterns |
| `visualize_sc_results.py` | ASCII visualization | Terminal-friendly plots |
| `visualize_trace_confidence.py` | Confidence graphs | Track evolution of confidence |
| `run_sc_experiments.sh` | Automated runs | Multiple configurations |

### `examples/` - Usage Examples

| Example | Purpose | Shows |
|---------|---------|-------|
| `example_offline.py` | Offline mode | Batch generation, all voting methods |
| `example_online.py` | Online mode | Confidence-based early stopping |

### `docs/` - Documentation

| Document | Audience | Content |
|----------|----------|---------|
| `QUICKSTART.md` | Everyone | TL;DR commands and examples |
| `SELF_CONSISTENCY.md` | SC users | Complete SC guide for AIME25 |
| `TRACE_VISUALIZATION.md` | Researchers | Confidence tracking guide |
| `IMPLEMENTATION.md` | Developers | Technical implementation details |
| `CHANGELOG.md` | All | Recent features and changes |
| `PROJECT_ORGANIZATION.md` | Contributors | This file |

## Usage Patterns

### Running Scripts

All scripts should be run from the **project root**:

```bash
# ✓ Correct (from root)
python scripts/run_traditional_sc_aime25.py --num_traces 64

# ✗ Wrong (from scripts/ directory)
cd scripts/
python run_traditional_sc_aime25.py --num_traces 64  # Import errors!
```

### Importing the Package

```python
# From anywhere in the project
from deepconf import DeepThinkLLM, prepare_prompt, equal_func

# Or specific imports
from deepconf.utils import weighted_majority_vote
from deepconf.outputs import DeepThinkOutput
```

### Reading Documentation

1. **Start here**: `README.md` (main overview)
2. **Quick commands**: `docs/QUICKSTART.md`
3. **Deep dive**: Relevant doc in `docs/`

## Output Directories

Generated during experiments (not in git):

```
outputs_sc/                        # SC experiment results
├── traditional_sc_aime25_detailed_*.json   # Full results
├── traditional_sc_aime25_summary_*.csv     # Spreadsheet
├── traditional_sc_aime25_stats_*.json      # Statistics
├── trace_confidence_qidN_*.pkl             # Saved trace data
├── trace_confidence_qidN_*.json            # Trace summary
└── trace_confidence_plot_*.png             # Visualization graphs
```

## Navigation Guide

### "I want to run self-consistency on AIME25"

1. Read: [`docs/QUICKSTART.md`](QUICKSTART.md)
2. Run: `python scripts/run_traditional_sc_aime25.py --num_traces 64`
3. Analyze: `python scripts/analyze_sc_results.py outputs_sc/*.json`

### "I want to visualize trace confidence"

1. Read: [`docs/TRACE_VISUALIZATION.md`](TRACE_VISUALIZATION.md)
2. Run: `python scripts/visualize_trace_confidence.py --qid 0 --num_traces 16`

### "I want to understand the implementation"

1. Read: [`docs/IMPLEMENTATION.md`](IMPLEMENTATION.md)
2. Check: `deepconf/wrapper.py` for main logic
3. Check: `deepconf/utils.py` for voting methods

### "I want to use this in my own code"

1. Read: [`examples/example_offline.py`](../examples/example_offline.py)
2. Import: `from deepconf import DeepThinkLLM, prepare_prompt`
3. Adapt: Copy example and modify for your use case

## Design Principles

### Why This Structure?

1. **Separation of Concerns**
   - `deepconf/`: Core library (reusable)
   - `scripts/`: Experiments (executable)
   - `docs/`: Documentation (readable)
   - `examples/`: Tutorials (learning)

2. **Easy Navigation**
   - All scripts in one place (`scripts/`)
   - All docs in one place (`docs/`)
   - Clear, descriptive names

3. **No Clutter**
   - Root level: Only essentials (README, requirements, setup)
   - Everything else: Organized in subdirectories

4. **Run from Root**
   - All paths relative to project root
   - Consistent imports
   - No confusion about working directory

### File Naming Conventions

- **Scripts**: Verb-based (`run_*.py`, `analyze_*.py`, `visualize_*.py`)
- **Docs**: Topic-based (`SELF_CONSISTENCY.md`, `TRACE_VISUALIZATION.md`)
- **Modules**: Noun-based (`wrapper.py`, `utils.py`, `outputs.py`)

## Migration from Old Structure

### What Changed?

**Before** (messy):
```
./
├── README.md
├── README_SC_AIME25.md
├── QUICK_START_SC.md
├── SC_AIME25_INDEX.md
├── IMPLEMENTATION_SUMMARY.md
├── TRACE_VISUALIZATION_GUIDE.md
├── NEW_FEATURES.md
├── run_traditional_sc_aime25.py
├── test_sc_single_question.py
├── analyze_sc_results.py
├── visualize_sc_results.py
├── visualize_trace_confidence.py
├── run_sc_experiments.sh
├── requirements_sc.txt
└── ...
```

**After** (clean):
```
./
├── README.md
├── requirements.txt
├── scripts/           # All scripts
├── docs/              # All documentation
├── deepconf/          # Core package
└── examples/          # Usage examples
```

### Path Updates

If you have old commands, update them:

```bash
# Old
python run_traditional_sc_aime25.py --num_traces 64
python analyze_sc_results.py outputs_sc/*.json

# New
python scripts/run_traditional_sc_aime25.py --num_traces 64
python scripts/analyze_sc_results.py outputs_sc/*.json
```

### Documentation Updates

| Old File | New File |
|----------|----------|
| `QUICK_START_SC.md` | `docs/QUICKSTART.md` |
| `README_SC_AIME25.md` | `docs/SELF_CONSISTENCY.md` |
| `TRACE_VISUALIZATION_GUIDE.md` | `docs/TRACE_VISUALIZATION.md` |
| `IMPLEMENTATION_SUMMARY.md` | `docs/IMPLEMENTATION.md` |
| `NEW_FEATURES.md` | `docs/CHANGELOG.md` |
| `SC_AIME25_INDEX.md` | Removed (info now in main README) |
| `requirements_sc.txt` | `requirements.txt` |

## Benefits of This Organization

### For Users

- ✅ Clear entry point: `README.md`
- ✅ Easy to find scripts: All in `scripts/`
- ✅ Easy to find docs: All in `docs/`
- ✅ No confusion about what to run

### For Developers

- ✅ Clean imports from `deepconf` package
- ✅ Easy to add new scripts (just add to `scripts/`)
- ✅ Easy to add new docs (just add to `docs/`)
- ✅ Clear separation: library vs experiments

### For Repository

- ✅ Professional appearance
- ✅ Standard Python project structure
- ✅ Easy to navigate on GitHub
- ✅ Clear for collaborators

## Contributing

When adding new files:

1. **New script?** → Add to `scripts/`
2. **New documentation?** → Add to `docs/`
3. **New core feature?** → Add to `deepconf/`
4. **New example?** → Add to `examples/`

Update `README.md` to link to your new content!

---

**Questions about organization?** See main [`README.md`](../README.md) or check [`docs/`](.)
