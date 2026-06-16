---
name: code-diff-learning
description: Analyze semantic code changes against predefined optimization ideas and record matching patterns (ignore renames, formatting, and equivalent refactors)
---

## What I do

Analyze **two versions of code** (original vs optimized) to determine whether the changes align with **predefined optimization ideas or patterns**.

Changes that are **semantically equivalent or idea-neutral** (e.g., variable renaming, formatting, equivalent refactors) are **explicitly ignored** and must **not** produce pattern matches.

The output serves as validation of optimization ideas and pattern accumulation for DSL generation and kernel evolution.

---

## When to use me

Use this skill when you have a pair of implementations (DSL / AscendC / Torch / mixed) and want to:

- Validate whether code changes **implement a specific optimization idea**.
- Identify **pattern matches** against predefined optimization concepts.
- Accumulate **pattern instances** that confirm or refute optimization ideas.

If the two versions are **semantically equivalent** or changes don't match any known patterns, this skill will explicitly conclude that **no matching patterns found**.

---

## Workflow

1. **Input Collection**
   - Receive **Original Code** and **Optimized Code**.
   - Receive **Optimization Idea(s)** to validate against.
   - (Optional) Receive human-written notes.
   - (Optional) Receive context hints:
     - canonical kernel name
     - target hardware
     - shapes / dtype

2. **Equivalence & Relevance Filtering (Critical Gate)**
   - Enumerate all observable differences.
   - Classify each difference into:
     - **(A) Idea-Neutral / Semantically Equivalent** → MUST IGNORE
     - **(B) Candidate for Idea Matching** → MUST ANALYZE
   - If **all** differences are (A):
     - Conclude early:
       > "The code changes are semantically equivalent or idea-neutral. No pattern matches found."
     - Do **not** proceed to further analysis.
     - Do **not** generate pattern records.

3. **Idea-Relevant Analysis**
   For each **(B)** change:
   - Check against provided **Optimization Idea(s)**:
     - Does this change implement the idea?
     - Which aspect of the idea does it address?
   - Identify affected dimension(s):
     - **Memory behavior** (GM/L2/L1/UB traffic, layout, reuse, alignment)
     - **Parallelism & scheduling** (core partitioning, vectorization, pipeline)
     - **Computation structure** (tiling, blocking, unrolling, reduction)
     - **Control flow & workload** (branches, loop bounds, executed work)
   - Explain the **mechanism**:
     - What structure changed?
     - How does it relate to the target idea?

4. **Structured Analysis Output**
   - Produce a structured analysis containing:
     - Idea validation results
     - Idea-relevant semantic differences only
     - Pattern generalization
   - If no matching patterns exist, produce only the verdict section.

5. **Knowledge Card Storage**
   - For each change that matches an optimization idea, call:
     ```
     knowledge_cards_create_knowledge_card_from_diff
     ```
   - Parameters: code_diff_entry, session_id, step_index, optional (profiling_data, kernel_info, scene_info)
   - If no matching changes are found, no knowledge card is created.

---

## Requirements & Constraints

### Analysis Granularity

This skill operates at **three mandatory levels**:

1. **Code-Level**
   What structural or semantic elements changed (ignoring naming and formatting).

2. **Mechanism-Level**
   How the change affects execution, memory access, or scheduling structure.

3. **Pattern-Level**
   How the change maps to the target optimization idea.

All three levels MUST be present in the output.

### Changes to Ignore (Must Not Produce Pattern Matches)

Treat the following as **no meaningful difference**:

- Variable, function, or type renaming
- Formatting, whitespace, or comment-only changes
- Reordering independent statements without changing dependencies
- Refactors that preserve:
  - control flow
  - data flow
  - memory access pattern
  - asymptotic executed work
- Algebraic or syntactic rewrites that do not change structure

If only the above changes exist, conclude:
> **No idea-matching pattern detected.**

### Idea-Relevant Changes (Must Analyze)

Only analyze differences that affect at least one of the following:

#### Memory Behavior
- GM/L2/L1/UB access pattern changes
- Buffering, staging, or reuse strategy changes
- Data layout, alignment, or padding changes

#### Parallelism & Scheduling
- Core or block partitioning changes
- Vectorization strategy changes
- Pipeline structure changes

#### Computation Structure
- Tiling, blocking, or unrolling changes
- Reduction strategy changes

#### Control Flow & Workload
- Branching that alters executed paths
- Loop bound or iteration order changes
- Changes that alter executed operations


## Input

The skill requires the following inputs:

### Required
- **Original Code** – The baseline DSL code before optimization
- **Optimized Code** – The DSL code after applying optimizations
- **Optimization Ideas** – A list of optimization ideas to validate against, each containing:
  - `idea_id` – unique identifier for the idea
  - `description` – what the optimization idea aims to achieve
  - `target_category` – memory / parallel / compute / control

### Optional
- **kernel_name** – canonical operator name (inferred if not provided)
- **target_hardware** – target hardware (e.g., Ascend 910B)
- **shapes/dtype** – tensor shapes and data types for context
- **human_notes** – additional context or notes from the user

## Output

### Knowledge Card Creation

When code changes match optimization ideas, call the MCP tool to store the knowledge card:

```
knowledge_cards_create_knowledge_card_from_diff
```

**Required parameters:**
- `code_diff_entry`: { kernel_name, target_idea, change_category, diff_summary, diff_detail }
- `session_id`: Unique session ID (e.g., `sess_{op}_{YYYYMMDD}_{seq}`)
- `step_index`: Current step index (0=baseline, 1,2,...=optimization steps)

**Optional parameters:**
- `profiling_data`: { bottleneck_type, task_duration_us: {before, after}, confidence }
- `kernel_info`: { category, compute_type }
- `scene_info`: { shape_signature, dtype, data_tags, task_tags }
- `runtime_info`: { device, dsl_before, dsl_after, profiling_path }

**Output:** The returned knowledge card includes: case_id, kernel, thought, scene, effect, chain, runtime.

If no matching changes are found, no knowledge card is created.
