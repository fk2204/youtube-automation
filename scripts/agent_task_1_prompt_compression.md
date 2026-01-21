# Agent Task 1: Compress Script Prompts in script_writer.py

## Objective
Reduce SCRIPT_PROMPT_TEMPLATE from 4200 characters to ~1000 characters (62% token reduction) while maintaining quality.

## File Location
C:\Users\fkozi\youtube-automation\src\content\script_writer.py

## Specific Changes Required

### 1. Lines 789-1000: Compress Main Script Prompt Template
Current: ~800 lines of verbose instructions
Target: ~300 lines with essential guidance only

**What to Keep:**
- Core requirements: duration, topic, style, audience
- Hook formula (1-2 examples only, not 5)
- Micro-payoffs concept (brief, 1-2 lines)
- Open loops concept (brief, 1-2 lines)
- Essential YouTube retention tactics

**What to Remove:**
- Redundant explanations
- Multiple examples of same concept
- Overly detailed instructions that AI can infer
- Verbose descriptions

### 2. Lines 1109-1299: Compress Niche Guides
Current: 3 niche guides with extensive detail (~190 lines)
Target: Essential differentiators only (~60 lines)

**For each niche (finance, psychology, storytelling):**
- Keep: CPM range, 1-2 hook examples, tone
- Remove: Lengthy explanations, multiple examples, redundant tactics
- Focus: What makes THIS niche different from others

### 3. Implementation Strategy
- Use concise bullet points instead of paragraphs
- Remove filler words and redundancy
- Keep technical terms but remove explanations
- Trust the AI model to understand context

### 4. Quality Validation
After compression, ensure:
- All 3 niches still have distinct personalities
- Hook formulas are clear (1 example each)
- Retention tactics are mentioned but not over-explained
- Template variables {duration}, {topic}, {style}, {audience} remain

## Expected Outcome
- SCRIPT_PROMPT_TEMPLATE: 4200 chars → ~1000 chars (75% reduction)
- NICHE_GUIDES: Compressed to essential differentiators only
- Code remains functional with same API
- No other changes to the file

## Testing
Run this command to verify it still works:
```bash
cd "C:\Users\fkozi\youtube-automation"
python -c "from src.content.script_writer import ScriptWriter; w = ScriptWriter(); print('OK')"
```

## Deliverable
Modified C:\Users\fkozi\youtube-automation\src\content\script_writer.py with compressed prompts.
