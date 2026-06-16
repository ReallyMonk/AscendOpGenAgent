---
name: ideapool-learner
description: Generate optimization strategy by combining IdeaPool, human knowledge, and knowledge cards. First identifies main ideas, then provides execution plan.
---

## What I do

I generate a comprehensive optimization strategy by combining:
1. **IdeaPool references** - prior kernel optimization patterns
2. **Human knowledge** - expert optimization knowledge from references/
3. **Knowledge cards** - stored optimization cases from MCP server

The output is a prioritized optimization strategy with:
- **Main optimization ideas** - key directions to pursue
- **Execution plan** - specific actionable steps for each idea

## When to use me

Use this skill when you need to plan optimization for a target kernel.

Typical use cases:
- Generate optimization direction before starting kernel evolution
- Combine multiple knowledge sources for comprehensive strategy
- Need both high-level ideas and specific execution steps

## Input

Required:
- target kernel name
- current DSL code (for analysis)

Optional:
- profiling data (if available)
- qualifiers: grad, quant, masked, dynamic, fused, inplace
- operator family

## Reference Locations

### IdeaPool
- `.opencode/skills/ideapool_learner/references/`
- Primary file: `expert_ideas.md`

### Human Knowledge
- `references/human_knowledge/`

### Knowledge Cards (MCP)
- Query via `knowledge_cards_search_knowledge_card_summaries`

## Workflow

### Step 1: Query Knowledge Cards (MCP)

Call `knowledge_cards_search_knowledge_card_summaries` to retrieve relevant past optimization cases:
- `kernel_name`: target kernel (required)
- `category`: operator category (optional)
- `effect_label`: effective/partial (optional, to find successful optimizations)
- `target_bottleneck`: UB_pressure, GM_bandwidth, pipeline_bubble, core_underutil (optional)
- `compute_type`: memory_bound, compute_bound, pipeline_bound (optional)
- `limit`: 5-10

If results found, for each case:
- Note the case_id for potential follow-up
- Extract: previous optimization ideas that worked, thought patterns, effectiveness
- Use `knowledge_cards_get_knowledge_card_by_id` to get full details if needed

### Step 2: Read IdeaPool References

Follow existing matching logic to find relevant IdeaPool references:
1. Exact match
2. Near-name variant
3. Same operator family
4. Similar computation pattern

Extract optimization ideas from `expert_ideas.md` files.

### Step 3: Scan Human Knowledge

Search `references/human_knowledge/` for:
- Kernel-specific optimization patterns
- General optimization principles
- Known issues and solutions

### Step 4: Analyze Current DSL

Analyze the current DSL code to identify:
- Current bottlenecks
- Optimization opportunities
- Code structure patterns

### Step 5: Synthesize Strategy

Combine all knowledge sources to generate optimization strategy:

**First: Identify Main Optimization Ideas**
Rank by:
- Expected impact (high/medium/low)
- Confidence from past results
- Applicability to current DSL

**Then: Provide Execution Plan**
For each main idea, specify:
- What to change
- Where to apply (specific code locations)
- Expected benefit
- Risk level

## MCP Tools

### 1. Search Knowledge Cards (Primary)
```
knowledge_cards_search_knowledge_card_summaries
```
Search for relevant past optimization cases.
- `kernel_name`: target operator name
- `category`: operator category (activation, normalization, matmul, conv, elementwise, reduction, attention, other)
- `compute_type`: bottleneck type (memory_bound, compute_bound, pipeline_bound, cube_underutil)
- `effect_label`: effectiveness (effective, partial, ineffective, negative)
- `session_id`: specific session to query
- `thought_id`: specific optimization idea ID
- `target_bottleneck`: UB_pressure, GM_bandwidth, pipeline_bubble, core_underutil
- `limit`: max results (default 20)

### 2. Get Knowledge Card by ID
```
knowledge_cards_get_knowledge_card_by_id
```
Get full knowledge card after finding interesting cases from search.
- `case_id`: specific case ID (e.g., case_LeakyReLU_sess_20260305_01_1)

### 3. Query Knowledge Cards (Advanced)
```
knowledge_cards_query_knowledge_cards
```
Advanced query with multiple filters.
- `kernel_name`, `category`, `compute_type`, `effect_label`, `session_id`, `thought_id`
- `shape_signature`: tensor shape pattern
- `dtype`: data type
- `step_index`: specific optimization step

### 4. List All Knowledge Cards
```
knowledge_cards_list_knowledge_cards
```
List all available knowledge cards.

### 5. Get Card Count
```
knowledge_cards_get_knowledge_card_count
```
Get total count and counts by category.

### 6. Get Schema
```
knowledge_cards_get_knowledge_card_schema
```
Get the JSON schema for knowledge cards.

## Output Format

### Target
- Kernel: <target kernel>
- Normalized: <normalized form>
- Current DSL: <path or summary>

### Knowledge Sources
- IdeaPool references: <list>
- Human knowledge: <list>
- Knowledge cards: <list of case_ids>

### Main Optimization Ideas

#### 1. <Primary Idea>
- **Impact**: <high/medium/low>
- **Source**: <knowledge card ID / IdeaPool reference / human knowledge>
- **Rationale**: <why this is the primary direction>

#### 2. <Secondary Idea>
- **Impact**: <high/medium/low>
- **Source**: <source>
- **Rationale**: <reasoning>

### Execution Plan

#### Idea 1: <Name>
| Step | Action | Location | Expected Benefit |
|------|--------|----------|------------------|
| 1 | <specific change> | <code location> | <description> |
| 2 | <specific change> | <code location> | <description> |

#### Idea 2: <Name>
| Step | Action | Location | Expected Benefit |
|------|--------|----------|------------------|
| 1 | <specific change> | <code location> | <description> |

### Priority
- **Try First**: <idea 1>, <idea 2>
- **Defer**: <idea n>
