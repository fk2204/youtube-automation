# YouTube Automation Agent System

A collection of specialized, token-efficient AI agents for the YouTube automation pipeline. Each agent is designed with a single responsibility and optimized for cost-effectiveness.

## Architecture Overview

```
                    +-------------------+
                    |  Agent Dispatcher |
                    |   (run.py agent)  |
                    +--------+----------+
                             |
         +-------------------+-------------------+
         |         |         |         |         |
    +----v----+ +--v---+ +---v--+ +----v---+ +---v----+
    | Research| |Quality| | SEO  | |Analytics| |Maintain|
    |  Agent  | | Agent | | Agent| |  Agent  | | Agent  |
    +---------+ +-------+ +------+ +---------+ +--------+
         |         |         |         |           |
    +----v---------v---------v---------v-----------v----+
    |              Shared Utilities                      |
    |  - TokenTracker (cost management)                 |
    |  - PromptCache (avoid duplicate calls)            |
    |  - CostOptimizer (provider selection)             |
    +---------------------------------------------------+
```

## Design Principles

### 1. Token Efficiency
- **Use smaller models for simple tasks**: Groq/Ollama for research, summaries, tags
- **Use powerful models sparingly**: Claude/GPT-4 only for complex script writing
- **Cache responses**: Avoid duplicate API calls with `PromptCache`
- **Batch operations**: Process multiple items in single requests

### 2. Single Responsibility
Each agent does ONE thing well:
- Research Agent: Find topics (does NOT write scripts)
- Quality Agent: Validate content (does NOT fix it automatically)
- SEO Agent: Optimize metadata (does NOT upload videos)

### 3. Reusability
- Agents work across sessions via CLI
- State persisted in SQLite databases
- Configuration via YAML files
- Results saved as JSON for later use

### 4. Self-Documenting
- Every agent logs its actions
- Token usage tracked per operation
- Results include reasoning and confidence scores

---

## Agent Catalog

## 1. Research Agent

**Purpose:** Find trending topics, analyze competitors, generate video ideas.

**When to use:**
- Before creating new content
- Weekly content planning sessions
- Exploring new niches
- Competitive analysis

**Token cost:** LOW (web search + summarize)
- Typical: 500-1,500 tokens per research session
- Provider: Groq (free) or Ollama (local)

**Input required:**
- `niche`: Content niche (e.g., "finance", "psychology")
- `channel`: Optional channel ID for niche auto-detection
- `count`: Number of ideas to generate (default: 5)

**Output produced:**
```json
{
  "ideas": [
    {
      "title": "5 Money Mistakes Costing You $1000/Year",
      "score": 87,
      "trend_score": 90,
      "competition_score": 75,
      "engagement_score": 85,
      "reasoning": "Loss aversion titles have highest CTR...",
      "keywords": ["money", "mistakes", "savings"]
    }
  ],
  "trends": [...],
  "metadata": {
    "niche": "finance",
    "generated_at": "2026-01-18T10:30:00",
    "tokens_used": 1250
  }
}
```

**Files it reads:**
- `config/channels.yaml` - Channel niche mapping
- `data/prompt_cache.db` - Cached research results

**Files it modifies:**
- `data/research_results.json` - Saves research output
- `data/token_usage.db` - Logs token usage

**Example invocations:**
```bash
# Find trending finance topics
python run.py agent research "find trending finance topics"

# Research for specific channel
python run.py agent research --channel money_blueprints

# Generate 10 viral ideas
python run.py agent research --niche psychology --count 10

# Competitor analysis
python run.py agent research --analyze-competitors --niche storytelling
```

---

## 2. Script Quality Agent

**Purpose:** Validate scripts, check engagement patterns, ensure best practices.

**When to use:**
- After script generation (before TTS)
- When reviewing existing scripts
- Quality assurance before upload
- A/B testing script variations

**Token cost:** MEDIUM
- Validation only: 200-500 tokens (rule-based, minimal AI)
- Full analysis with AI: 1,000-2,000 tokens
- Provider: Groq for quick checks, Claude for deep analysis

**Input required:**
- `script_file`: Path to script file OR script text
- `niche`: Content niche for context-specific validation
- `is_short`: Whether validating a YouTube Short

**Output produced:**
```json
{
  "score": 85,
  "is_valid": true,
  "checks": {
    "hook_quality": {"passed": true, "score": 90, "details": "Strong curiosity trigger"},
    "title_viral": {"passed": true, "score": 85, "details": "Contains numbers and power words"},
    "retention_structure": {"passed": true, "score": 80, "details": "3 open loops detected"},
    "cta_placement": {"passed": false, "score": 60, "details": "CTA too early (15s mark)"}
  },
  "recommendations": [
    "Move CTA to 30% mark (around 90 seconds)",
    "Add one more micro-payoff in first 30 seconds"
  ],
  "metadata": {
    "niche": "finance",
    "is_short": false,
    "tokens_used": 450
  }
}
```

**Files it reads:**
- Script file provided by user
- `src/utils/best_practices.py` - Validation rules
- `docs/COMPETITOR_ANALYSIS.md` - Best practices reference

**Files it modifies:**
- `data/quality_reports/` - Saves validation reports
- `data/token_usage.db` - Logs token usage

**Example invocations:**
```bash
# Quick validation (rule-based, no AI)
python run.py agent quality "output/script.txt" --quick

# Full AI analysis
python run.py agent quality "output/script.txt" --niche finance --full

# Validate as YouTube Short
python run.py agent quality "output/short_script.txt" --short

# Get improvement suggestions
python run.py agent quality "output/script.txt" --suggest-improvements
```

---

## 3. SEO Optimizer Agent

**Purpose:** Optimize titles, descriptions, tags for YouTube discoverability.

**When to use:**
- Before uploading videos
- Refreshing metadata on existing videos
- A/B testing titles
- Keyword research

**Token cost:** LOW
- Title optimization: 300-600 tokens
- Full metadata: 800-1,200 tokens
- Provider: Groq (free)

**Input required:**
- `title`: Current video title
- `description`: Current description
- `tags`: Current tags (optional)
- `niche`: Content niche
- `transcript`: Video transcript (optional, for better tag generation)

**Output produced:**
```json
{
  "optimized": {
    "title": "5 Money Mistakes Costing You $1,000/Year (2026)",
    "description": "Stop losing money to these common mistakes...\n\n00:00 Introduction\n00:30 Mistake #1...",
    "tags": ["money mistakes", "personal finance", "save money", "budgeting tips", "financial literacy"],
    "hashtags": ["#MoneyTips", "#PersonalFinance", "#Budgeting"]
  },
  "changes": {
    "title": "Added year, specific dollar amount",
    "description": "Added timestamps, expanded keyword coverage",
    "tags": "Reordered by search volume, removed generic terms"
  },
  "seo_score": {
    "before": 65,
    "after": 88
  },
  "metadata": {
    "tokens_used": 750
  }
}
```

**Files it reads:**
- `src/utils/best_practices.py` - SEO patterns
- `config/channels.yaml` - Channel-specific keywords

**Files it modifies:**
- `data/seo_optimizations/` - Saves optimization history
- `data/token_usage.db` - Logs token usage

**Example invocations:**
```bash
# Optimize title only
python run.py agent seo --title "How to save money"

# Full metadata optimization
python run.py agent seo --file "output/video_metadata.json" --niche finance

# Generate tags from transcript
python run.py agent seo --transcript "output/transcript.txt" --generate-tags

# A/B test titles
python run.py agent seo --ab-test "output/video_metadata.json" --variations 3
```

---

## 3b. SEO Strategist Agent (World-Class)

**Purpose:** Comprehensive SEO strategy including keyword research, intent analysis, competitor monitoring, CTR prediction, A/B testing, and content calendar planning.

**When to use:**
- Before creating content (keyword research, intent analysis)
- Content planning sessions (content calendar, topic suggestions)
- Pre-upload optimization (full metadata optimization, A/B variants)
- Competitive analysis (gap identification, pattern extraction)

**Token cost:** ZERO to LOW
- Keyword research: 0 tokens (pytrends + YouTube autocomplete)
- Competitor analysis: 0 tokens (rule-based pattern extraction)
- CTR prediction: 0 tokens (learned patterns)
- A/B variant generation: 0-500 tokens (rule-based with optional AI)
- Full strategy: 0-1,000 tokens (mostly cached)
- Provider: Local/Groq

**Components:**
- `KeywordResearcher`: Research keywords using pytrends + YouTube autocomplete
- `SearchIntentAnalyzer`: Classify search intent (informational, navigational, transactional, commercial)
- `CompetitorAnalyzer`: Analyze top YouTube results, extract patterns, find content gaps
- `PerformancePredictor`: Predict CTR and retention based on metadata patterns
- `ABTestManager`: Generate and score title variants using psychological triggers
- `ContentCalendar`: SEO-driven content planning with topic suggestions

**Input required:**
- `keyword`: Keyword for research (research, competitors commands)
- `niche`: Content niche (finance, psychology, storytelling)
- `title`: Title for optimization/A/B testing
- `file`: JSON file with metadata (optimize command)

**Output produced:**
```json
{
  "success": true,
  "operation": "research:passive income",
  "data": {
    "keyword_data": {
      "keyword": "passive income",
      "opportunity_score": 75.0,
      "competition_level": "medium",
      "youtube_autocomplete": ["passive income ideas", "passive income 2026", ...],
      "rising_queries": ["ai passive income", "digital products", ...]
    },
    "intent": {
      "primary_intent": "informational",
      "confidence": 0.85,
      "content_recommendations": ["Use step-by-step structure", ...]
    },
    "competitor_report": {
      "common_title_patterns": ["[Number] [Keyword] That [Benefit]", ...],
      "content_gaps": ["Beginner's guide to passive income", ...]
    }
  },
  "recommendations": [
    "Search intent: informational (85% confidence)",
    "Competition level: medium",
    "Opportunity score: 75/100",
    "Rising queries: ai passive income, digital products"
  ]
}
```

**Files it reads:**
- `src/agents/seo_data/keyword_cache.db` - Cached keyword research (7-day TTL)
- `src/agents/seo_data/patterns.json` - Learned successful patterns
- `src/agents/seo_data/test_results.json` - A/B test performance history

**Files it modifies:**
- `src/agents/seo_data/keyword_cache.db` - Caches new research
- `src/agents/seo_data/competitor_data.json` - Saves competitor analysis
- `src/agents/seo_data/test_results.json` - Records A/B test results

**Example invocations:**
```bash
# Keyword research with trends and competition
python run.py agent seo-strategist research "passive income" --niche finance

# Full content strategy with topic suggestions
python run.py agent seo-strategist strategy --niche psychology --topics 10

# Optimize metadata before upload
python run.py agent seo-strategist optimize --file output/metadata.json

# Generate A/B test title variants
python run.py agent seo-strategist ab-test "My Video Title" --variants 5

# Competitor analysis
python run.py agent seo-strategist competitors "money mistakes" --top 10

# Generate content calendar
python run.py agent seo-strategist calendar --niche finance --weeks 4
```

---

## 4. Analytics Agent

**Purpose:** Analyze video performance, identify patterns, suggest improvements.

**When to use:**
- Weekly performance reviews
- Identifying underperforming content
- Finding successful patterns
- Planning content strategy

**Token cost:** LOW-MEDIUM
- Basic analysis: 500-1,000 tokens
- Deep insights: 1,500-2,500 tokens
- Provider: Groq for summaries, Claude for strategic insights

**Input required:**
- `channel`: Channel ID or YouTube channel URL
- `period`: Analysis period (7d, 30d, 90d)
- `metrics`: Specific metrics to analyze (views, retention, CTR)

**Output produced:**
```json
{
  "summary": {
    "total_views": 125000,
    "avg_retention": 45.2,
    "avg_ctr": 8.5,
    "top_performer": "5 Money Mistakes...",
    "underperformer": "Basic Budgeting Tips"
  },
  "patterns": {
    "best_posting_time": "15:00 UTC",
    "best_title_pattern": "Number + Loss Aversion",
    "optimal_length": "8-12 minutes",
    "high_retention_topics": ["mistakes", "secrets", "psychology"]
  },
  "recommendations": [
    "Post more content about 'mistakes' - 35% higher retention",
    "Avoid generic titles - 40% lower CTR",
    "Consider longer videos (12+ min) for storytelling niche"
  ],
  "metadata": {
    "period": "30d",
    "videos_analyzed": 15,
    "tokens_used": 1200
  }
}
```

**Files it reads:**
- `data/video_performance.db` - Historical performance data
- `data/upload_history.json` - Upload timestamps and metadata
- YouTube Analytics API (if configured)

**Files it modifies:**
- `data/analytics_reports/` - Saves analysis reports
- `data/token_usage.db` - Logs token usage

**Example invocations:**
```bash
# Weekly channel review
python run.py agent analytics --channel money_blueprints --period 7d

# Find patterns across all channels
python run.py agent analytics --all-channels --period 30d

# Analyze specific video
python run.py agent analytics --video "dQw4w9WgXcQ" --deep

# Generate content strategy
python run.py agent analytics --strategy --niche finance
```

---

## 5. Maintenance Agent

**Purpose:** Code health checks, dependency updates, cleanup tasks.

**When to use:**
- Weekly maintenance windows
- Before major updates
- After errors occur
- System health monitoring

**Token cost:** MINIMAL (mostly local operations)
- Health check: 0 tokens (local)
- Error analysis: 300-800 tokens
- Provider: Local for most tasks, Groq for error analysis

**Input required:**
- `task`: Maintenance task type (health, cleanup, deps, errors)
- `fix`: Whether to auto-fix issues (default: false)

**Output produced:**
```json
{
  "health": {
    "status": "healthy",
    "issues": [],
    "warnings": ["Token usage approaching daily limit"]
  },
  "cleanup": {
    "files_removed": 15,
    "space_freed": "250 MB",
    "old_logs_archived": 7
  },
  "dependencies": {
    "outdated": ["moviepy: 1.0.3 -> 2.0.0"],
    "vulnerabilities": [],
    "recommendation": "Safe to update"
  },
  "errors": {
    "recent_count": 3,
    "patterns": ["TTS timeout (2 occurrences)", "API rate limit (1 occurrence)"],
    "suggestions": ["Implement retry logic for TTS", "Add rate limit backoff"]
  },
  "metadata": {
    "last_check": "2026-01-18T10:00:00",
    "tokens_used": 0
  }
}
```

**Files it reads:**
- `logs/*.log` - Application logs
- `requirements.txt` - Dependencies
- `data/*.db` - Database health
- `output/` - Generated files

**Files it modifies:**
- `logs/` - Archives old logs
- `output/` - Cleans up temp files
- `data/maintenance_reports/` - Saves reports

**Example invocations:**
```bash
# Full health check
python run.py agent maintenance --health

# Cleanup old files
python run.py agent maintenance --cleanup --days 30

# Check dependencies
python run.py agent maintenance --deps

# Analyze recent errors
python run.py agent maintenance --errors --suggest-fixes
```

---

## Quick Reference

| Agent | Token Cost | Provider | Primary Use |
|-------|------------|----------|-------------|
| Research | LOW | Groq/Ollama | Find trending topics |
| Quality | MEDIUM | Groq/Claude | Validate scripts |
| SEO | LOW | Groq | Optimize metadata |
| Analytics | LOW-MED | Groq/Claude | Performance analysis |
| Maintenance | MINIMAL | Local/Groq | System health |

---

## Token Optimization Strategies

### 1. Provider Selection by Task

```python
# Automatic provider selection based on task complexity
from src.utils.token_manager import get_cost_optimizer

optimizer = get_cost_optimizer(daily_budget=10.0)

# Simple tasks -> Groq (free)
provider = optimizer.select_provider("idea_generation")  # Returns "groq"

# Complex tasks -> Claude (when budget allows)
provider = optimizer.select_provider("script_full", prefer_quality=True)  # Returns "claude"
```

### 2. Response Caching

```python
from src.utils.token_manager import get_prompt_cache

cache = get_prompt_cache()

# Check cache before API call
cached = cache.get("Find trending finance topics 2026")
if cached:
    return cached  # 0 tokens!

# Otherwise, make API call and cache
response = ai.generate(prompt)
cache.set(prompt, response)
```

### 3. Batch Processing

```python
# Instead of 5 separate API calls:
for topic in topics:
    ai.generate(f"Analyze {topic}")  # 5 API calls = ~5,000 tokens

# Use one batched call:
ai.generate(f"Analyze these 5 topics: {topics}")  # 1 API call = ~2,000 tokens
```

### 4. Progressive Enhancement

```python
# Start with cheap validation
result = quality_agent.quick_check(script)  # Rule-based, 0 tokens

if result.needs_review:
    # Only use AI for edge cases
    result = quality_agent.full_analysis(script)  # AI-powered, ~1,500 tokens
```

---

## Integration with Existing Pipeline

### Video Creation Flow

```
1. python run.py agent research --niche finance
   └── Saves ideas to data/research_results.json

2. python run.py video money_blueprints
   └── Uses best idea from research
   └── Generates script
   └── Creates video

3. python run.py agent quality "output/script.txt"
   └── Validates before upload
   └── Returns recommendations

4. python run.py agent seo --file "output/metadata.json"
   └── Optimizes title, description, tags

5. [Upload to YouTube]

6. python run.py agent analytics --channel money_blueprints
   └── Tracks performance
   └── Feeds back into research
```

### Scheduled Agent Tasks

Add to `config/channels.yaml`:

```yaml
agent_schedule:
  research:
    frequency: weekly
    day: Sunday
    time: "10:00"
  analytics:
    frequency: weekly
    day: Monday
    time: "09:00"
  maintenance:
    frequency: daily
    time: "03:00"
```

---

## Error Handling

All agents implement consistent error handling:

```python
try:
    result = agent.run(params)
except TokenBudgetExceeded:
    # Switch to free provider
    result = agent.run(params, provider="ollama")
except RateLimitError:
    # Exponential backoff
    time.sleep(60)
    result = agent.run(params)
except Exception as e:
    # Log and save partial results
    logger.error(f"Agent failed: {e}")
    agent.save_partial_results()
```

---

## Adding New Agents

To create a new agent:

1. Create `src/agents/<agent_name>_agent.py` using the template
2. Add to `src/agents/__init__.py`
3. Register in `run.py` agent dispatcher
4. Document in this file
5. Add tests in `tests/test_agents.py`

Template structure:

```python
"""
<Agent Name> Agent
<Brief description>
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from loguru import logger
from ..utils.token_manager import get_token_manager, get_cost_optimizer

@dataclass
class AgentResult:
    success: bool
    data: Dict[str, Any]
    tokens_used: int
    error: Optional[str] = None

class NewAgent:
    """Agent description."""

    TASK_COMPLEXITY = "simple"  # simple, medium, complex

    def __init__(self, provider: str = None):
        self.tracker = get_token_manager()
        self.optimizer = get_cost_optimizer()
        self.provider = provider or self.optimizer.select_provider(self.TASK_COMPLEXITY)

    def run(self, **kwargs) -> AgentResult:
        """Main agent entry point."""
        pass
```

---

## Monitoring and Costs

### Daily Cost Report

```bash
python run.py cost
```

Output:
```
       TOKEN USAGE & COST REPORT
==================================================

Period          Input Tokens   Output Tokens      Cost
-------------------------------------------------------
Today                  5,250          3,100   $0.0042
This Week             35,000         21,000   $0.0285
This Month           142,000         89,000   $0.1156

Provider        Input Tokens   Output Tokens      Cost
-------------------------------------------------------
groq                 120,000         75,000   $0.0000
ollama                22,000         14,000   $0.0000
claude                    0              0   $0.0000

--- Summary ---
Average cost per video: $0.0019
Daily budget: $10.00 | Spent: $0.0042 | Remaining: $9.9958
```

### Budget Alerts

Agents automatically check budget before expensive operations:

```python
if self.tracker.check_budget()["warning"]:
    logger.warning("Budget 80% depleted, switching to free providers")
    self.provider = "groq"
```
