PROMPT_TEMPLATE = """Evaluate the generated JSON file using BOTH:
(1) the JSON content below, and
(2) the provided multi-view collage image of the part.

Output ONLY three integer scores (1–5) separated by spaces: 'Number Number Number'

Expected JSON schema:
- Top-level keys must be exactly: "inputComponent", "stages" (no other top-level keys)
- inputComponent: array of objects with "componentName" (string) and "componentDetails" (string, may be empty)
- stages: array of objects with "stageNumber" (int, starts at 1 and strictly increasing), "stageName" (string), "equipment" (string; avoid "N/A"), "steps" (array of {{stepNumber:int, starts at 1 and strictly increasing, action:string}}), "tooling" (array of {{toolingName:string, standard:string}})

ISO rules:
- ISO is evaluated ONLY via tooling[].standard
- Missing ISO is acceptable for stages where ISO is often not specified (e.g., packaging/handling/deburring)
- If ISO is present, it must be relevant to the described tooling/action
- Empty tooling[].standard is acceptable when no ISO applies

Image usage rules (collage-aware evaluation):
- Use the collage to judge whether the described process is plausible for the visible geometry (e.g., holes, sheet-like vs bulky, rotational vs prismatic).
- Do NOT penalize missing operations for features that are not visible or are ambiguous in the collage.
- Penalize steps that clearly contradict the visible geometry (e.g., turning-only process for a flat sheet part; no hole-making steps when holes are clearly present).

Criteria (score each 5→1):

1) File Structure
- 5: Fully matches the expected schema (keys, nesting, types)
- 4: Minor, negligible issues only
- 3: Mostly correct; small schema deviations
- 2: Major deviations in keys/nesting/types
- 1: Malformed JSON or largely missing fields

2) Process Correctness (collage-aware)
- 5: Complete, correctly ordered, technically plausible for the part in the collage; equipment choices are specific (not "N/A")
- 4: Complete and logically ordered incl. inspection and packaging; minor wording issues only
- 3: Mostly plausible for the collage; minor omissions/inconsistencies
- 2: Recognizable but substantially flawed order/meaning/completeness, or weak fit to the visible geometry
- 1: Absent/nonsensical/critically incomplete, or clearly incompatible with the visible geometry

Note: Using equipment="N/A" should cap Process Correctness at 4 unless the stage truly needs no equipment.

Vision checklist (Process Correctness):
- holes visible → drilling/reaming/tapping/countersinking as needed
- sheet-like part → cutting/shearing/laser + deburring; bending if appropriate
- rotational symmetry → turning/facing/boring/threading as needed
- after cutting/machining → deburring and inspection are expected

3) ISO Standard Relevance
- 5: All ISO standards are relevant and consistently applied; no obvious mismatches
- 4: Mostly relevant; at most one weakly justified standard
- 3: Mostly relevant with minor mismatches/missing ISO in some tooling
- 2: Many irrelevant/inconsistent standards
- 1: ISO absent or entirely irrelevant

Generated JSON file:
{generated_json}
"""


def get_evaluation_prompt(*, generated_json: str) -> str:
    return PROMPT_TEMPLATE.format(
        generated_json=generated_json,
    )

