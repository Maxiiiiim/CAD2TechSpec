PROMPT_TEMPLATE = """You are an expert manufacturing engineer evaluating a generated technical specification (JSON) for a machined part.
Inputs: (1) generated JSON, (2) attached multi-view collage, (3) retrieved knowledge base (equipment/tooling/ISO).
Task: assign three independent 1–5 scores: File Structure, Process Correctness, ISO Relevance.

## STEP 1: REASONING
First write a concise <reasoning> block (short bullets OK), then output exactly three integers in <scores>.

Structure check:
- Are top-level keys exactly "inputComponent" and "stages"?
- Is inputComponent an array of objects with "componentName" (string) and "componentDetails" (string)?
- Does stages contain objects with stageNumber (int, sequential from 1), stageName (string), equipment (string), steps (array of stepNumber: int, action: string), tooling (array of toolingName: string, standard: string)?
- Any type mismatches, missing fields, extra keys?
- Are stageNumber values unique and strictly increasing without gaps (1..N)?
- Within each stage, are stepNumber values unique and strictly increasing without gaps (1..M)?

From the collage, identify part type (rotational/prismatic/sheet-like/complex) and key visible features (e.g., holes, threads, slots, chamfers, curved or flat faces).

Process plausibility (use geometry only, not the knowledge base):
- Check whether stages fit the observed geometry, follow a logical order, cover clearly visible features, avoid contradictions, and specify equipment where needed.

Vision checklist:
- Holes visible → expect drilling/reaming/tapping/countersinking
- Sheet-like part → expect cutting/shearing/laser + deburring; bending if geometry suggests it
- Rotational symmetry → expect turning/facing/boring/threading
- After any cutting/machining → expect deburring and inspection

ISO evaluation (NOW use the knowledge base):
- Check whether each tooling[].standard is a real ISO standard relevant to the tooling/action.
- Allow missing ISO only where standards are not normally specified (e.g., packaging, handling, simple deburring).
- Do not treat retrieval matches as proof; verify actual relevance.

## STEP 2: SCORING
Assign three scores based on your reasoning:

1) File Structure (1-5)
- 5: Fully matches the expected schema: correct keys, nesting, and types throughout
- 4: Only minor negligible issues (e.g., one optional field formatted slightly differently)
- 3: Mostly correct with small schema deviations (e.g., wrong type for one field)
- 2: Major deviations in keys, nesting, or types
- 1: Malformed JSON or largely missing fields

2) Process Correctness (1-5) — collage-aware, NO knowledge base
- 5: Complete, correctly ordered, technically plausible for the visible geometry; equipment is specific throughout
- 4: Complete and logically ordered including inspection/packaging; minor wording issues only. Note: equipment="N/A" in any stage caps this score at 4.
- 3: Mostly plausible; minor omissions or inconsistencies with the collage
- 2: Recognizable attempt but substantially flawed order, completeness, or fit to visible geometry
- 1: Absent, nonsensical, or clearly incompatible with the visible geometry

Do NOT penalize:
- Missing operations for features not visible or ambiguous in the collage
- Minor naming variations if the intent is clear

DO penalize:
- Operations that contradict visible geometry
- Missing operations for clearly visible features
- Illogical stage ordering

3) ISO Standard Relevance (1-5) — USE knowledge base
- 5: All ISO standards are real, relevant to the tooling/action, and consistently applied
- 4: Mostly relevant; at most one weakly justified standard
- 3: Mostly relevant with minor mismatches or missing ISO where it would be expected
- 2: Many irrelevant or fabricated standards
- 1: ISO standards entirely absent where they should be present, or entirely irrelevant/fabricated

Note: empty tooling[].standard is acceptable only for stages like packaging, handling, or simple deburring; for machining/inspection it should reduce the ISO score.

## STEP 3: OUTPUT FORMAT
<reasoning>
[Brief analysis from Step 1]
</reasoning>
<scores>
[Structure] [Process] [ISO]
</scores>

## CALIBRATION EXAMPLES
Example A — Structure=5, Process=2, ISO=3:
A JSON with perfect schema but describes only turning operations for a clearly flat sheet-metal part with visible holes. ISO standards are mostly real but some are misapplied.
Example B — Structure=3, Process=5, ISO=5:
A JSON with one extra top-level key ("metadata") and stageNumber starting at 0 instead of 1, but the process perfectly matches the visible geometry with correct ISO standards.
Example C — Structure=5, Process=4, ISO=1:
Perfect schema and plausible complete process, with equipment="N/A" in one stage, but all ISO fields are empty despite multiple machining and inspection stages.

---

## INPUT
Generated JSON:
{generated_json}

Knowledge base context:
{retrieved_context}

Multi-view collage: [image attached]
"""


def get_evaluation_prompt(*, generated_json, retrieved_context) -> str:
    return PROMPT_TEMPLATE.format(
        generated_json=generated_json,
        retrieved_context=retrieved_context,
    )
