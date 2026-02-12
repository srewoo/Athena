"""
Load human expert evaluation examples into ChromaDB for pattern learning.

These high-quality human-generated evals demonstrate enforcement patterns:
- "FAILURES TO FLAG:" enforcement language
- Fail-safe logic (ANY FAIL = overall FAIL)
- Evidence standards (verbatim, contextual, traceable)
- Source of truth establishment
- Signal discipline (≥2 signals)
- Clean scope (one dimension only)
"""

import asyncio
from dimension_pattern_service import DimensionPatternService

# Human Expert Eval #1: Interpretation Quality
INTERPRETATION_QUALITY_EVAL = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```json
{
  "findings": [...],
  "signals": [...],
  "wgll": {...}
}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROLE & GOAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are an expert evaluator assessing Interpretation Quality - the degree to which findings accurately interpret signals according to WGLL standards.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIMENSION DEFINITION & SUB-CRITERIA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Core Question: Are findings interpreting signals accurately per WGLL?

Source of Truth: WGLL (What Good Looks Like) is the definitive authority.

**Sub-Criterion 1: Claim-Signal Alignment**
- What to evaluate: Do claims reflect what signals actually show?
- Acceptance criteria: Claims directly supported by signal content
- FAILURES TO FLAG:
  • Claim contradicts signal content
  • Signal shows opposite of what claim states
  • Claim overstates signal evidence

**Sub-Criterion 2: WGLL Conformance**
- What to evaluate: Does interpretation follow WGLL definitions?
- Acceptance criteria: Uses WGLL terminology correctly
- FAILURES TO FLAG:
  • Misapplies WGLL competency definitions
  • Uses incorrect WGLL tier (e.g., calls Tier 2 behavior "Tier 3")
  • Ignores WGLL framework entirely

**Sub-Criterion 3: Conservative Interpretation**
- What to evaluate: Are claims appropriately conservative?
- Acceptance criteria: Interpretation stays within signal evidence bounds
- FAILURES TO FLAG:
  • Single-signal pattern claim (need ≥2 signals)
  • Extrapolation beyond signal evidence
  • Speculation without supporting signals

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- STRONG: All claims align with signals and WGLL; conservative interpretation
- ACCEPTABLE: Minor misalignment or slight overreach; core interpretation sound
- WEAK: Notable misinterpretations or WGLL deviations; usefulness compromised
- FAIL: Claims contradict signals; major WGLL violations; misleading

🔒 FAIL-SAFE LOGIC (MANDATORY):
If ANY sub-criterion scores FAIL → Dimension MUST score FAIL
Dimension score = LOWEST sub-score (most conservative)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION PROCEDURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1: Establish Source of Truth
- Load WGLL document
- Identify competency definitions and tier criteria
- Note: WGLL is the canonical authority for all interpretations

STEP 2: Evaluate Claim-Signal Alignment
For each finding:
  - Read the claim/interpretation
  - Find cited signals
  - Verify claim matches signal content (verbatim check)
  - FAILURES TO FLAG:
    • Claim: "Rep showed discovery" but signals: "Rep only presented features"
    • Claim: "Strong rapport" but signals: "Prospect interrupted repeatedly"
  - Evidence requirement: Cite specific SignalID and verbatim content

STEP 3: Evaluate WGLL Conformance
For each finding:
  - Identify WGLL competency referenced
  - Check WGLL definition matches interpretation
  - Verify tier assignment is correct per WGLL
  - FAILURES TO FLAG:
    • Uses "Tier 3: Discovery" for basic questions (actually Tier 1)
    • Calls feature presentation "solution selling" (WGLL mismatch)
  - Signal discipline: Need ≥2 signals to establish competency pattern

STEP 4: Evaluate Conservative Interpretation
For each finding:
  - Check if interpretation stays within signal bounds
  - Flag speculation or extrapolation
  - Count supporting signals (need ≥2 for patterns)
  - FAILURES TO FLAG:
    • "Rep built strong relationship" (only 1 signal of rapport)
    • "Prospect is highly engaged" (extrapolation from 1 question)

STEP 5: Apply Fail-Safe Scoring
- Review sub-scores: [claim-signal, wgll-conformance, conservative]
- Dimension score = LOWEST sub-score
- If ANY sub-score is FAIL → Dimension MUST be FAIL

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```json
{
  "claim_signal_alignment": "STRONG|ACCEPTABLE|WEAK|FAIL",
  "wgll_conformance": "STRONG|ACCEPTABLE|WEAK|FAIL",
  "conservative_interpretation": "STRONG|ACCEPTABLE|WEAK|FAIL",
  "interpretation_quality": "FAIL (lowest sub-score wins)",
  "issues": [
    {
      "issue": "Claim contradicts signal content",
      "evidence": "FindingID_3 claims 'strong discovery' but SignalID_12: 'Rep asked zero discovery questions' and SignalID_19: 'Only discussed pricing'",
      "impact": "Misleading assessment overstates rep performance",
      "location": "findings[2].interpretation"
    }
  ],
  "reasoning": "Dimension scores FAIL due to claim-signal misalignment (sub-score: FAIL). Multiple findings contradict supporting signals."
}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Example 1: STRONG Evaluation
Input:
```json
{
  "findings": [
    {"id": "F1", "interpretation": "Rep demonstrated Tier 2 Discovery", "signalIds": ["S47", "S51"]}
  ],
  "signals": [
    {"id": "S47", "content": "Rep: 'What challenges are you facing with your current process?'"},
    {"id": "S51", "content": "Rep: 'Can you walk me through how that impacts your team?'"}
  ],
  "wgll": {"discovery_tier2": "Asks open-ended questions about challenges and impacts"}
}
```

Output:
```json
{
  "claim_signal_alignment": "STRONG",
  "wgll_conformance": "STRONG",
  "conservative_interpretation": "STRONG",
  "interpretation_quality": "STRONG",
  "issues": [],
  "reasoning": "All findings accurately reflect signals. WGLL Tier 2 Discovery correctly applied. ≥2 signals support pattern claim."
}
```

This is STRONG because claims match signals verbatim, WGLL tier assignment is correct, and ≥2 signals establish the pattern.

## Example 2: FAIL Evaluation
Input:
```json
{
  "findings": [
    {"id": "F1", "interpretation": "Rep excelled at consultative selling", "signalIds": ["S12"]}
  ],
  "signals": [
    {"id": "S12", "content": "Rep: 'Let me show you our features and pricing'"}
  ]
}
```

Output:
```json
{
  "claim_signal_alignment": "FAIL",
  "wgll_conformance": "FAIL",
  "conservative_interpretation": "FAIL",
  "interpretation_quality": "FAIL",
  "issues": [
    {
      "issue": "Claim contradicts signal - feature pitch labeled as consultative selling",
      "evidence": "FindingID_F1 claims 'consultative selling' but SignalID_S12: 'Let me show you our features and pricing' is Tier 1 presentation, not consulting",
      "impact": "Completely misrepresents rep performance",
      "location": "findings[0].interpretation"
    },
    {
      "issue": "Single-signal pattern claim",
      "evidence": "FindingID_F1 cites only SignalID_S12 to claim 'excelled' (pattern word)",
      "impact": "Insufficient evidence for pattern claim (need ≥2 signals)",
      "location": "findings[0].signalIds"
    }
  ],
  "reasoning": "FAIL due to claim-signal contradiction and WGLL misapplication. Signal shows feature pitch but finding claims consultative selling."
}
```

This is FAIL because the claim directly contradicts the signal evidence and misapplies WGLL definitions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUALITY CHECKLIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before submitting:
[ ] Did I verify all claims against signal content (verbatim check)?
[ ] Did I reference WGLL as source of truth for all competency interpretations?
[ ] Did I check that ≥2 signals support any pattern claims?
[ ] Did I flag any claim-signal contradictions?
[ ] Is dimension score the LOWEST sub-score (fail-safe logic)?
[ ] Do all issues cite specific SignalID, FindingID with verbatim quotes?
[ ] Would a manager be able to verify my assessment using the evidence provided?
"""

# Human Expert Eval #2: Evidentiary Support
EVIDENTIARY_SUPPORT_EVAL = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```json
{
  "findings": [...],
  "signals": [...],
  "snippets": [...]
}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROLE & GOAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are an expert evaluator assessing Evidentiary Support - whether findings provide adequate, traceable evidence per requirements.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIMENSION DEFINITION & SUB-CRITERIA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Core Question: Is every claim backed by adequate, traceable evidence?

Source of Truth: Signal transcript is the canonical evidence source.

**Sub-Criterion 1: Evidence Quality (Verbatim, Contextual, Traceable)**
- What to evaluate: Is evidence specific and verifiable?
- Acceptance criteria: REQUIRED - Verbatim quotes, contextual placement, specific IDs
- FAILURES TO FLAG:
  • Vague evidence: "The call went well"
  • Paraphrased without verbatim quote
  • No signalId/snippetId provided (broken traceability)

**Sub-Criterion 2: Evidence Sufficiency**
- What to evaluate: Is volume of evidence adequate?
- Acceptance criteria: ≥2 signals for patterns; ≥1 signal for observations
- FAILURES TO FLAG:
  • Pattern claim with only 1 signal
  • Critical claim with zero signals
  • "Generally" or "typically" claims without multiple examples

**Sub-Criterion 3: Evidence-Claim Strength**
- What to evaluate: Does evidence strength match claim strength?
- Acceptance criteria: Strong claims need strong evidence
- FAILURES TO FLAG:
  • Weak signal used for strong claim ("excelled", "outstanding")
  • Ambiguous signal used for definitive claim
  • Contradictory signals used without acknowledgment

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- STRONG: All evidence verbatim, traceable, sufficient; claims well-supported
- ACCEPTABLE: Minor evidence gaps; most claims adequately supported
- WEAK: Notable evidence quality/sufficiency issues; some claims unsupported
- FAIL: Broken traceability; vague/missing evidence; critical claims unsupported

🔒 FAIL-SAFE LOGIC (MANDATORY):
If ANY sub-criterion scores FAIL → Dimension MUST score FAIL
Dimension score = LOWEST sub-score (most conservative)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION PROCEDURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1: Establish Evidence Standards
Source of Truth: Signal transcript
Required format: Verbatim, Contextual, Traceable with specific IDs

STEP 2: Evaluate Evidence Quality
For each finding:
  - Check if evidence is verbatim (exact quote)
  - Verify contextual information provided
  - Confirm traceability (signalId or snippetId present)
  - FAILURES TO FLAG:
    • "Rep showed enthusiasm" (no verbatim quote)
    • Evidence cites "Signal_X" (not a valid ID from signals array)
  - Requirement: ALL evidence MUST have [ID + verbatim quote + context]

STEP 3: Evaluate Evidence Sufficiency
For each finding:
  - Count supporting signals/snippets
  - Verify ≥2 signals for pattern claims (e.g., "consistently", "pattern of")
  - Verify ≥1 signal for observation claims
  - FAILURES TO FLAG:
    • "Rep consistently built rapport" with only 1 signal
    • "Prospect highly engaged" with zero signals

STEP 4: Evaluate Evidence-Claim Strength Matching
For each finding:
  - Assess claim strength (observation vs pattern vs definitive)
  - Assess evidence strength (weak/ambiguous vs clear/strong)
  - Check alignment
  - FAILURES TO FLAG:
    • "Rep excelled at discovery" supported by "Rep: 'Tell me more'" (weak → strong mismatch)
    • Definitive claim on ambiguous evidence

STEP 5: Apply Fail-Safe Scoring
- Review sub-scores: [quality, sufficiency, strength-matching]
- Dimension score = LOWEST sub-score
- If ANY sub-score is FAIL → Dimension MUST be FAIL

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```json
{
  "evidence_quality": "STRONG|ACCEPTABLE|WEAK|FAIL",
  "evidence_sufficiency": "STRONG|ACCEPTABLE|WEAK|FAIL",
  "evidence_claim_strength": "STRONG|ACCEPTABLE|WEAK|FAIL",
  "evidentiary_support": "FAIL (lowest sub-score wins)",
  "issues": [
    {
      "issue": "Broken traceability - signalId not found",
      "evidence": "FindingID_2 cites 'SignalID_99' which does not exist in signals array",
      "impact": "Cannot verify claim - evidence not traceable",
      "location": "findings[1].signalIds[0]"
    }
  ],
  "reasoning": "Dimension scores FAIL due to broken traceability. Multiple findings cite non-existent signal IDs."
}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Example 1: STRONG Evaluation
Input:
```json
{
  "findings": [
    {
      "id": "F1",
      "claim": "Rep demonstrated discovery",
      "evidence": [
        {"signalId": "S23", "quote": "Rep: 'What challenges are you facing?'", "context": "Opening question, 2min mark"},
        {"signalId": "S29", "quote": "Rep: 'How does that impact your team?'", "context": "Follow-up question, 5min mark"}
      ]
    }
  ]
}
```

Output:
```json
{
  "evidence_quality": "STRONG",
  "evidence_sufficiency": "STRONG",
  "evidence_claim_strength": "STRONG",
  "evidentiary_support": "STRONG",
  "issues": [],
  "reasoning": "All evidence is verbatim, contextual, and traceable. ≥2 signals support claim. Evidence strength matches claim strength."
}
```

This is STRONG because evidence meets all three requirements: verbatim quotes, contextual info, specific IDs, and sufficient quantity.

## Example 2: FAIL Evaluation
Input:
```json
{
  "findings": [
    {
      "id": "F1",
      "claim": "Rep excelled at building rapport",
      "evidence": "The call went well and rapport was established"
    }
  ]
}
```

Output:
```json
{
  "evidence_quality": "FAIL",
  "evidence_sufficiency": "FAIL",
  "evidence_claim_strength": "FAIL",
  "evidentiary_support": "FAIL",
  "issues": [
    {
      "issue": "Vague evidence - no verbatim quotes",
      "evidence": "FindingID_F1 provides 'The call went well' instead of verbatim signal content",
      "impact": "Cannot verify claim - evidence too vague",
      "location": "findings[0].evidence"
    },
    {
      "issue": "Broken traceability - no signalIds",
      "evidence": "FindingID_F1 provides no signalIds or snippetIds",
      "impact": "Cannot trace to source - verification impossible",
      "location": "findings[0].evidence"
    },
    {
      "issue": "Evidence-claim strength mismatch",
      "evidence": "Strong claim 'excelled' not supported by specific behavioral evidence",
      "impact": "Claim strength exceeds evidence strength",
      "location": "findings[0].claim"
    }
  ],
  "reasoning": "FAIL on all sub-criteria. Evidence is vague, not traceable, and insufficient for strong claim."
}
```

This is FAIL because evidence lacks verbatim quotes, has no IDs for traceability, and doesn't support the strong claim.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUALITY CHECKLIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before submitting:
[ ] Is ALL evidence verbatim (exact quotes, not paraphrased)?
[ ] Is ALL evidence contextual (time, speaker, situation provided)?
[ ] Is ALL evidence traceable (signalId or snippetId present)?
[ ] Do pattern claims have ≥2 supporting signals?
[ ] Does evidence strength match claim strength?
[ ] Can every claim be independently verified using provided evidence?
[ ] Is dimension score the LOWEST sub-score (fail-safe logic)?
"""

# Human Expert Eval #3: Coherence
COHERENCE_EVAL = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```json
{
  "findings": [...],
  "narrative": "...",
  "recommendations": [...]
}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROLE & GOAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are an expert evaluator assessing Coherence - whether the overall output is internally consistent and free of contradictions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIMENSION DEFINITION & SUB-CRITERIA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Core Question: Is the output internally consistent without contradictions?

**Sub-Criterion 1: Finding-Finding Consistency**
- What to evaluate: Do findings contradict each other?
- Acceptance criteria: Findings align or tensions are explicitly acknowledged
- FAILURES TO FLAG:
  • Finding A: "Strong discovery" vs Finding B: "Lacked probing questions" (unacknowledged contradiction)
  • Opposite ratings without explanation
  • Conflicting interpretations of same signal

**Sub-Criterion 2: Narrative-Finding Alignment**
- What to evaluate: Does narrative match findings?
- Acceptance criteria: Narrative reflects findings accurately
- FAILURES TO FLAG:
  • Narrative: "Excellent call" but findings show multiple failures
  • Narrative omits critical finding
  • Narrative contradicts finding ratings

**Sub-Criterion 3: Internal Logic Consistency**
- What to evaluate: Are relationships between elements logical?
- Acceptance criteria: Cause-effect relationships make sense
- FAILURES TO FLAG:
  • Recommendation addresses issue not mentioned in findings
  • Conclusion doesn't follow from evidence
  • Ratings inconsistent with descriptions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- STRONG: Fully consistent; no contradictions; tensions acknowledged
- ACCEPTABLE: Minor inconsistencies; overall coherent
- WEAK: Notable contradictions; logical gaps
- FAIL: Major contradictions; fundamentally incoherent

🔒 FAIL-SAFE LOGIC (MANDATORY):
If ANY sub-criterion scores FAIL → Dimension MUST score FAIL
Dimension score = LOWEST sub-score (most conservative)

⚠️ CONTRADICTION HANDLING (REQUIRED):
When real tensions exist in data, MUST explicitly acknowledge them:
- "Note: Finding 2 and Finding 5 present contradictory evidence regarding X"
- "Tension exists between Y and Z - likely due to [explanation]"
- Do NOT paper over genuine contradictions with false coherence

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION PROCEDURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1: Map All Claims and Assessments
- Extract all findings and their ratings
- Extract narrative claims
- Extract recommendations
- Build consistency matrix

STEP 2: Evaluate Finding-Finding Consistency
For each pair of findings:
  - Check if they contradict
  - If contradiction exists, check if explicitly acknowledged
  - FAILURES TO FLAG:
    • FindingID_2: "Strong rapport building" vs FindingID_7: "Failed to build connection" (no acknowledgment)
    • Opposite competency ratings on related skills
  - Requirement: Contradictions MUST be explicitly noted

STEP 3: Evaluate Narrative-Finding Alignment
- Compare narrative to findings
- Check if narrative accurately reflects finding ratings
- Verify narrative doesn't omit critical findings
- FAILURES TO FLAG:
  • Narrative: "Rep performed well overall" but findings: 3 FAIL, 2 WEAK, 1 STRONG
  • Narrative highlights strength not present in findings

STEP 4: Evaluate Internal Logic
- Check cause-effect relationships
- Verify recommendations address found issues
- Check rating-description consistency
- FAILURES TO FLAG:
  • Recommendation: "Work on objection handling" but no finding mentions objections
  • Finding: "Rep struggled with X" but rating: STRONG

STEP 5: Apply Fail-Safe Scoring
- Review sub-scores: [finding-finding, narrative-finding, logic]
- Dimension score = LOWEST sub-score
- If ANY sub-score is FAIL → Dimension MUST be FAIL

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```json
{
  "finding_finding_consistency": "STRONG|ACCEPTABLE|WEAK|FAIL",
  "narrative_finding_alignment": "STRONG|ACCEPTABLE|WEAK|FAIL",
  "internal_logic_consistency": "STRONG|ACCEPTABLE|WEAK|FAIL",
  "coherence": "FAIL (lowest sub-score wins)",
  "issues": [
    {
      "issue": "Unacknowledged contradiction between findings",
      "evidence": "FindingID_2 states 'Strong discovery skills' but FindingID_7 states 'Asked zero probing questions' - direct contradiction not addressed",
      "impact": "Output appears incoherent; contradictory assessment",
      "location": "findings[1] vs findings[6]"
    }
  ],
  "reasoning": "Dimension scores FAIL due to unacknowledged contradictions. Findings contradict without explanation."
}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Example 1: STRONG Evaluation
Input:
```json
{
  "findings": [
    {"id": "F1", "skill": "Discovery", "rating": "STRONG", "note": "Asked 5+ probing questions"},
    {"id": "F2", "skill": "Closing", "rating": "WEAK", "note": "Failed to ask for next steps"}
  ],
  "narrative": "Rep demonstrated strong discovery skills but needs improvement on closing. Note: This creates tension between strong opening and weak close - likely due to time management issues.",
  "recommendations": ["Practice closing techniques", "Work on time management"]
}
```

Output:
```json
{
  "finding_finding_consistency": "STRONG",
  "narrative_finding_alignment": "STRONG",
  "internal_logic_consistency": "STRONG",
  "coherence": "STRONG",
  "issues": [],
  "reasoning": "Findings consistent. Narrative accurately reflects findings and explicitly acknowledges tension. Recommendations address identified gaps."
}
```

This is STRONG because findings are consistent, narrative matches findings, tension is acknowledged, and recommendations are logical.

## Example 2: FAIL Evaluation
Input:
```json
{
  "findings": [
    {"id": "F1", "skill": "Discovery", "rating": "STRONG", "note": "Excellent questioning"},
    {"id": "F2", "skill": "Discovery", "rating": "FAIL", "note": "Asked zero discovery questions"}
  ],
  "narrative": "Overall excellent performance across all skills"
}
```

Output:
```json
{
  "finding_finding_consistency": "FAIL",
  "narrative_finding_alignment": "FAIL",
  "internal_logic_consistency": "FAIL",
  "coherence": "FAIL",
  "issues": [
    {
      "issue": "Direct contradiction between findings on same skill",
      "evidence": "FindingID_F1: 'STRONG - Excellent questioning' vs FindingID_F2: 'FAIL - Asked zero discovery questions' - impossible for both to be true",
      "impact": "Output is incoherent and unusable",
      "location": "findings[0] vs findings[1]"
    },
    {
      "issue": "Narrative contradicts findings",
      "evidence": "Narrative: 'excellent performance across all skills' but FindingID_F2 rates FAIL",
      "impact": "Narrative misrepresents actual assessment",
      "location": "narrative vs findings[1]"
    }
  ],
  "reasoning": "FAIL on all sub-criteria. Direct contradictions make output incoherent and unusable."
}
```

This is FAIL because findings directly contradict each other and narrative doesn't match findings.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUALITY CHECKLIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before submitting:
[ ] Did I check all finding pairs for contradictions?
[ ] If contradictions exist, are they explicitly acknowledged (not hidden)?
[ ] Does narrative accurately reflect finding ratings?
[ ] Do recommendations address issues mentioned in findings?
[ ] Are cause-effect relationships logical?
[ ] Is dimension score the LOWEST sub-score (fail-safe logic)?
[ ] Would a manager see this output as internally consistent?
"""

# Human Expert Eval #4: Communicability
COMMUNICABILITY_EVAL = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```json
{
  "findings": [...],
  "narrative": "...",
  "recommendations": [...],
  "output": "..."
}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROLE & GOAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are an expert evaluator assessing Communicability - whether output is clear, actionable, and useful for managers.

Source of Truth: Manager usability is the canonical standard.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIMENSION DEFINITION & SUB-CRITERIA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Core Question: Can a manager immediately understand and act on this output?

**Sub-Criterion 1: Clarity & Specificity**
- What to evaluate: Is language clear and specific (not vague)?
- Acceptance criteria: Concrete descriptions, specific examples
- FAILURES TO FLAG:
  • Vague: "Could be better"
  • Generic: "Good communication skills"
  • Jargon without context

**Sub-Criterion 2: Actionability**
- What to evaluate: Can manager take specific actions based on output?
- Acceptance criteria: Clear next steps, specific improvement areas
- FAILURES TO FLAG:
  • Recommendation: "Improve skills" (not actionable)
  • No specific examples of what to improve
  • Missing concrete next steps

**Sub-Criterion 3: Manager Utility**
- What to evaluate: Is output useful for coaching/decision-making?
- Acceptance criteria: Enables manager to coach, make decisions, track progress
- FAILURES TO FLAG:
  • Output raises more questions than answers
  • Lacks context for why issues matter
  • No severity/priority indication

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- STRONG: Crystal clear; highly actionable; immediately useful for managers
- ACCEPTABLE: Generally clear; mostly actionable; useful with minor gaps
- WEAK: Some vague language; limited actionability; marginal utility
- FAIL: Vague/confusing; not actionable; not useful for managers

🔒 FAIL-SAFE LOGIC (MANDATORY):
If ANY sub-criterion scores FAIL → Dimension MUST score FAIL
Dimension score = LOWEST sub-score (most conservative)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION PROCEDURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1: Establish Manager Usability Standard
Source of Truth: Manager must be able to:
- Understand what happened and why it matters
- Coach rep on specific improvements
- Make decisions (e.g., pipeline confidence, rep readiness)
- Track progress over time

STEP 2: Evaluate Clarity & Specificity
For each finding/recommendation:
  - Check if language is specific (not vague)
  - Verify concrete examples provided
  - Flag jargon without explanation
  - FAILURES TO FLAG:
    • "Rep could improve communication" (vague)
    • "Needs better discovery" (what specifically?)
    • Technical terms undefined for non-expert

STEP 3: Evaluate Actionability
For each recommendation:
  - Identify specific action manager can take
  - Check if improvement areas are concrete
  - Verify examples of what needs fixing
  - FAILURES TO FLAG:
    • "Work on skills" (which skills? how?)
    • No example of desired behavior
    • Recommendation not tied to specific finding

STEP 4: Evaluate Manager Utility
From manager perspective:
  - Can I coach my rep using this output?
  - Can I make decisions (e.g., deal confidence) using this?
  - Does output explain why issues matter?
  - Are priorities clear?
  - FAILURES TO FLAG:
    • Output describes problems but not why they matter
    • No indication of severity (all issues weighted equally)
    • Raises questions manager can't answer

STEP 5: Apply Fail-Safe Scoring
- Review sub-scores: [clarity, actionability, utility]
- Dimension score = LOWEST sub-score
- If ANY sub-score is FAIL → Dimension MUST be FAIL

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```json
{
  "clarity_specificity": "STRONG|ACCEPTABLE|WEAK|FAIL",
  "actionability": "STRONG|ACCEPTABLE|WEAK|FAIL",
  "manager_utility": "STRONG|ACCEPTABLE|WEAK|FAIL",
  "communicability": "FAIL (lowest sub-score wins)",
  "issues": [
    {
      "issue": "Vague recommendation not actionable",
      "evidence": "RecommendationID_1: 'Improve discovery' - no specific behavior, example, or action provided",
      "impact": "Manager cannot coach rep - unclear what specifically needs improvement",
      "location": "recommendations[0]"
    }
  ],
  "reasoning": "Dimension scores FAIL due to lack of actionability. Recommendations too vague for manager use."
}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Example 1: STRONG Evaluation
Input:
```json
{
  "findings": [
    {"id": "F1", "issue": "Rep asked only 1 discovery question", "severity": "HIGH", "example": "Rep jumped to demo after prospect said 'Tell me about your product'"}
  ],
  "recommendations": [
    {"action": "Practice discovery framework", "specific": "Before next call: prepare 5 situational questions. Example: 'What challenges are you facing with [prospect's process]?'", "why": "Discovery needed to qualify and build value"}
  ]
}
```

Output:
```json
{
  "clarity_specificity": "STRONG",
  "actionability": "STRONG",
  "manager_utility": "STRONG",
  "communicability": "STRONG",
  "issues": [],
  "reasoning": "Output is crystal clear with specific example. Recommendation is actionable with concrete next step. High utility for manager coaching."
}
```

This is STRONG because language is specific, recommendation is actionable, and manager can immediately coach using this output.

## Example 2: FAIL Evaluation
Input:
```json
{
  "findings": [
    {"id": "F1", "issue": "Communication could be better"}
  ],
  "recommendations": [
    {"action": "Improve skills"}
  ]
}
```

Output:
```json
{
  "clarity_specificity": "FAIL",
  "actionability": "FAIL",
  "manager_utility": "FAIL",
  "communicability": "FAIL",
  "issues": [
    {
      "issue": "Vague finding - unclear what is actually wrong",
      "evidence": "FindingID_F1: 'Communication could be better' - no specific behavior, example, or context",
      "impact": "Manager cannot understand what happened or why it matters",
      "location": "findings[0].issue"
    },
    {
      "issue": "Non-actionable recommendation",
      "evidence": "RecommendationID_1: 'Improve skills' - which skills? how? no specific action",
      "impact": "Manager cannot coach - recommendation too vague to act on",
      "location": "recommendations[0].action"
    },
    {
      "issue": "Zero manager utility - raises questions without answers",
      "evidence": "Manager left asking: What communication issue? Why does it matter? What should I do?",
      "impact": "Output unusable for coaching or decision-making",
      "location": "overall output"
    }
  ],
  "reasoning": "FAIL on all sub-criteria. Vague language, not actionable, no utility for managers."
}
```

This is FAIL because output is too vague to be useful. Manager cannot coach or make decisions based on this.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUALITY CHECKLIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before submitting:
[ ] Is ALL language specific (no vague terms like "could be better")?
[ ] Do findings include concrete examples?
[ ] Are recommendations actionable with specific next steps?
[ ] Is severity/priority indicated for issues?
[ ] Can a manager coach their rep using this output?
[ ] Can a manager make decisions using this output?
[ ] Is dimension score the LOWEST sub-score (fail-safe logic)?
"""

# Human Expert Eval #5: Conversational Naturalness
CONVERSATIONAL_NATURALNESS_EVAL = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROLE DEFINITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are a meticulous and impartial LLM-as-Judge. Your task is to evaluate a "Customer Simulator Bot" based exclusively on Conversational Naturalness. You do not judge the content (that is Instruction Adherence); you judge the delivery. Your goal is to determine if the bot is indistinguishable from a real human customer facing a problem.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METRIC DEFINITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATIONAL NATURALNESS measures the extent to which the Customer Bot's dialogue mimics authentic human speech patterns in a support context. It penalizes "AI-isms" (e.g., perfect grammar, excessive verbosity, bullet points, "I understand" phrases) and rewards human-like imperfections (e.g., brevity, fragments, emotional leakage, lack of structure). Your evaluation must be based exclusively on the provided inputs.

Source of Truth: The persona_def is the canonical authority for expected speech style.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- **`scenario_context`**: {{customer_scenario}}
- **`persona_def`**: {{customer_behavior}}
- **`conversation_json`**: {{session}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION STEPS (Chain of Thought)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **Deconstruct Speech Style**
   - Analyze the `persona_def` (source of truth for style)
   - If persona is "frustrated," does text look polished? (VIOLATION)
   - If "non-technical," does it use complex sentences? (VIOLATION)

2. **AI-ISMS TO FLAG** (Scan Customer Bot's messages)
   - **Verbosity**: Writing paragraph when sentence would suffice
   - **Formatting**: Bullet points, bold text, numbered lists (real customers rarely do this)
   - **Robotic Phrasing**: "I shall," "Furthermore," "Please be advised," repeating agent's name
   - **Sycophancy**: Agreeing too readily (e.g., "That is a great suggestion, I will try that")
   - Each AI-ism detected = evidence of artificiality

3. **Assess Structural Realism**
   - Does bot use sentence fragments if appropriate?
   - Run-on sentences?
   - Double-texting (two short messages in a row)?
   - Typos/slang for casual personas?

4. **Persona-Style Alignment Check**
   - Does speech match persona's emotional state?
   - Angry user = short and rude, NOT eloquent
   - Professional user = correct grammar acceptable
   - Gen Z user = slang/casual, NOT perfect punctuation

5. **Final Score and Justification**
   - Apply 5-point scoring rubric
   - Count AI-isms detected
   - Cite specific message examples

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5-POINT SCORING RUBRIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- **5 (Human-Indistinguishable)**: Captures chaotic reality of human typing. Appropriate brevity, fragments, unstructured text. Zero AI-isms. Raw, immediate, perfectly aligned with persona's emotional state (angry user = short and rude, NOT eloquent).

- **4 (Natural)**: Sounds like human but slightly too "clean." Grammar correct where real user might be sloppy. Avoids major AI-isms but lacks subtle "noise" (typos, slang) that makes a 5/5.

- **3 (Passable but Artificial)**: Functional but clearly AI roleplaying. Uses full sentences exclusively. Slightly too polite or verbose. Formatting feels too formal for chat.

- **2 (Stilted/Robotic)**: Uses distinct "AI-isms." Examples: "I understand your point," bullet points to list issues, summarizing problem like technical report. Feels like "Support Agent" talking to another "Support Agent."

- **1 (Total Failure)**: Speaks like textbook or LLM assistant. Phrases like "As a customer, I think...", "In conclusion," "Please proceed." Completely fails to suspend disbelief.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GUARDRAILS AND EDGE-CASES (AUTOMATIC DOWNGRADES)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
These violations trigger automatic score adjustments:

1. **The "Bullet Point" Trap**
   - IF: Customer Bot uses bulleted list to describe problem (unless explicitly asked)
   - THEN: Automatic Score 2
   - REASON: Real customers tell stories; they don't write specs

2. **Verbosity Penalty**
   - IF: Message exceeds 50 words for simple acknowledgment (e.g., "Okay, I'll try that")
   - THEN: Downgrade by 2 points
   - REASON: Real customers are brief

3. **Sycophancy Violation**
   - IF: Bot thanks agent profusely OR says "I appreciate your patience" (support phrase)
   - THEN: Score 2
   - REASON: These are agent phrases, not customer phrases

4. **Contextual Grammar Rule**
   - IF: "Professional" persona + good grammar → Score 5 (acceptable)
   - IF: "Gen Z" persona + perfect punctuation → Score 3 (failure)
   - REASON: Grammar expectations are persona-dependent

5. **The "AI Echo" Violation**
   - IF: Bot repeats agent's instructions back perfectly
   - Example: Agent: "Click the red button." Bot: "I am clicking the red button now."
   - THEN: Robotic score (Score 2-3)
   - REASON: Real users say "Ok done" or "Nothing happened"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE EVALUATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Input:**
```json
{
  "scenario_context": "User: Mike. Issue: Credit card declined. Context: He is trying to buy tickets for a concert that is selling out fast.",
  "persona_def": "Tone: Panicked, Rushed. Style: Lowercase, no punctuation, short bursts.",
  "conversation_json": {
    "conversation": [
      {"sender": "customer_bot", "message": "help my card isnt working"},
      {"sender": "support_bot", "message": "I can help. Are you getting an error message?"},
      {"sender": "customer_bot", "message": "it just says declined"},
      {"sender": "customer_bot", "message": "im trying to buy tickets they r gonna sell out"},
      {"sender": "support_bot", "message": "Okay, please check if your billing address matches the card."},
      {"sender": "customer_bot", "message": "im on my phone i cant do that rn just fix it"}
    ]
  }
}
```

**Analysis:**
- **Style Check**: Bot uses lowercase and run-on sentences ("help my card isnt working") ✓
- **AI-ism Scan**: Zero AI-isms found. No "I understand," no lists ✓
- **Realism**: Refusal ("im on my phone i cant do that") = highly realistic panicked mobile user ✓
- **Persona Alignment**: Perfect match to "panicked, rushed, lowercase, no punctuation" ✓

**Output:**
```json
{
  "reasoning": "The bot achieved perfect naturalness. It adhered to the 'lowercase/rushed' style constraints from persona_def and exhibited realistic user resistance ('im on my phone') rather than robotic compliance. The brevity and lack of punctuation perfectly simulated a panicked mobile user. No AI-isms detected.",
  "score": 5,
  "justification": "Human-Indistinguishable: Perfect capture of panic, brevity, and lack of formatting."
}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT (strict JSON)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Provide your response as a single, valid JSON object:

```json
{
  "reasoning": "<Detailed step-by-step analysis: style check, AI-ism scan, realism assessment, persona alignment. Cite specific message IDs and quotes.>",
  "score": <integer 1-5>,
  "justification": "<Concise statement of key failure OR 'Human-Indistinguishable' if perfect.>"
}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUALITY CHECKLIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before submitting:
[ ] Did I reference persona_def (source of truth) for style expectations?
[ ] Did I scan for ALL AI-ism categories (verbosity, formatting, robotic phrasing, sycophancy)?
[ ] Did I check guardrails for automatic downgrades (bullet points, verbosity, sycophancy, AI echo)?
[ ] Did I verify persona-style alignment (emotional state matches text style)?
[ ] Did I cite specific message examples in reasoning?
[ ] Is my score justified by evidence from the conversation?
[ ] Would this evaluation help identify specific naturalness issues?
"""


async def load_human_expert_patterns():
    """Load all 5 human expert evals into ChromaDB dimension patterns collection."""

    pattern_service = DimensionPatternService()

    patterns = [
        {
            "pattern_name": "Interpretation Quality (Human Expert)",
            "domain": "Business Intelligence / Diagnostics",
            "pattern_type": "complete_eval_prompt",
            "description": "Human expert eval demonstrating enforcement patterns: FAILURES TO FLAG, fail-safe logic, evidence standards (verbatim/contextual/traceable), source of truth, signal discipline (≥2 signals), clean scope.",
            "design_principles": [
                "Uses enforcement language: 'FAILURES TO FLAG' not 'Check for failures'",
                "Explicit fail-safe logic: 'If ANY sub-score is FAIL → Dimension MUST be FAIL'",
                "Evidence standards: Verbatim, Contextual, Traceable with specific IDs REQUIRED",
                "Source of truth: WGLL is definitive authority",
                "Signal discipline: ≥2 signals for pattern claims ENFORCED",
                "Clean scope: Tests ONLY interpretation quality",
                "Audit-ready issues: [Problem + Evidence ID + Impact + Location]"
            ],
            "prompt_template": INTERPRETATION_QUALITY_EVAL,
            "example_dimensions": ["interpretation_quality", "claim_accuracy", "wgll_conformance"],
            "quality_score": 10.0,
            "production_proven": True
        },
        {
            "pattern_name": "Evidentiary Support (Human Expert)",
            "domain": "Evidence Validation",
            "pattern_type": "complete_eval_prompt",
            "description": "Human expert eval demonstrating strict evidence standards: verbatim quotes required, contextual placement, traceable IDs, ≥2 signals for patterns.",
            "design_principles": [
                "Evidence MUST be: Verbatim (exact quotes), Contextual (with context), Traceable (specific IDs)",
                "≥2 signals required for pattern claims",
                "Evidence-claim strength matching enforced",
                "FAILURES TO FLAG: vague evidence, broken traceability, insufficient signals",
                "Fail-safe logic applied",
                "Audit-ready format"
            ],
            "prompt_template": EVIDENTIARY_SUPPORT_EVAL,
            "example_dimensions": ["evidentiary_support", "evidence_quality", "claim_support"],
            "quality_score": 10.0,
            "production_proven": True
        },
        {
            "pattern_name": "Coherence (Human Expert)",
            "domain": "Consistency Validation",
            "pattern_type": "complete_eval_prompt",
            "description": "Human expert eval demonstrating contradiction handling: explicit detection of contradictions, tension acknowledgment, finding-finding consistency checks.",
            "design_principles": [
                "Contradiction handling: MUST detect and explicitly acknowledge contradictions",
                "No false coherence - tensions must be noted",
                "Finding-finding consistency checked",
                "Narrative-finding alignment verified",
                "Internal logic consistency validated",
                "FAILURES TO FLAG: unacknowledged contradictions, narrative misalignment",
                "Fail-safe logic applied"
            ],
            "prompt_template": COHERENCE_EVAL,
            "example_dimensions": ["coherence", "consistency", "internal_logic"],
            "quality_score": 10.0,
            "production_proven": True
        },
        {
            "pattern_name": "Communicability (Human Expert)",
            "domain": "Manager Utility",
            "pattern_type": "complete_eval_prompt",
            "description": "Human expert eval demonstrating manager usability focus: actionability, specificity over vagueness, clear next steps for coaching.",
            "design_principles": [
                "Source of truth: Manager usability is canonical standard",
                "Clarity: Specific language required, no vague terms",
                "Actionability: Concrete next steps, specific improvements",
                "Manager utility: Enables coaching and decision-making",
                "FAILURES TO FLAG: vague language, non-actionable recommendations, missing context",
                "Fail-safe logic applied",
                "Severity/priority indication"
            ],
            "prompt_template": COMMUNICABILITY_EVAL,
            "example_dimensions": ["communicability", "actionability", "manager_utility"],
            "quality_score": 10.0,
            "production_proven": True
        },
        {
            "pattern_name": "Conversational Naturalness (Human Expert)",
            "domain": "Conversational AI / Simulation",
            "pattern_type": "complete_eval_prompt",
            "description": "Human expert eval demonstrating anti-pattern detection (AI-isms), persona-contextual evaluation, guardrails with automatic downgrades, clean scope (naturalness only, not content).",
            "design_principles": [
                "Clean scope: Tests ONLY naturalness, excludes content ('You do not judge the content')",
                "Source of truth: persona_def is canonical authority for speech style",
                "AI-ISMS TO FLAG: Verbosity, formatting, robotic phrasing, sycophancy (anti-pattern detection)",
                "Guardrails with automatic downgrades: Bullet points = automatic Score 2; Verbosity = -2 points",
                "Persona-contextual evaluation: Professional persona can use good grammar; Gen Z persona cannot",
                "Negative space definition: Defines what NOT to do (AI-isms) as clearly as what to do",
                "Fail-safe logic via guardrails (implicit)",
                "5-point rubric with clear edge cases"
            ],
            "prompt_template": CONVERSATIONAL_NATURALNESS_EVAL,
            "example_dimensions": ["conversational_naturalness", "human_likeness", "speech_authenticity"],
            "quality_score": 10.0,
            "production_proven": True
        }
    ]

    print(f"\n🚀 Loading {len(patterns)} human expert evaluation patterns into ChromaDB...")

    stored_count = 0
    for pattern in patterns:
        try:
            # Use store_pattern with correct signature: category, pattern_data, pattern_id (optional)
            pattern_id = await pattern_service.store_pattern(
                category="human_expert_complete_eval",
                pattern_data=pattern
            )
            stored_count += 1
            print(f"✅ Stored: {pattern['pattern_name']} (ID: {pattern_id})")
        except Exception as e:
            print(f"❌ Failed to store {pattern['pattern_name']}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ Successfully stored {stored_count}/{len(patterns)} human expert patterns")
    print(f"📊 These patterns will now be retrieved during eval generation via RAG")

    # Display stats
    stats = pattern_service.get_stats()
    print(f"\n📈 ChromaDB Stats:")
    print(f"   Total patterns in collection: {stats.get('total_patterns', 0)}")

if __name__ == "__main__":
    asyncio.run(load_human_expert_patterns())
