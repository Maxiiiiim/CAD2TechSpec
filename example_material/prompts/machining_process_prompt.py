PROMPT_TEMPLATE = """Create a short and exact description of the technological process of manufacturing a product based on an image providing {n_dims} dimensions of a 3D model in JPG format. The process should be described in JSON format and include all stages of machining, starting from analysing the drawing and ending with obtaining the finished product.

The component has the following dimensions: {length} mm in length, {width} mm in width, and {height} mm in height. In addition, it includes {hole_count} hole type(s) with the following parameters: {holes_description}.
You may use the following processing stages as a baseline (not an exhaustive list): Cutting, Turning, Inspection, Milling, Bench work, Surface coating, CNC machining center operations, CNC milling, Electrical discharge machining (EDM), Shaping of polymer materials, Vertical milling, Bending, Shearing, Complete manufacturing according to process route card, Laser cutting.

The response should be JSON only, with no additional text. JSON example:

{{
    "inputComponent": [
        {{
            "componentName": "Name of the input component",
            "componentDetails": "Some details about the following input component if applicable"
        }}
    ],
    "stages": [
        {{
            "stageNumber": 1,
            "stageName": "Name of the stage #1",
            "equipment": "Name and/or model of equipment",
            "steps": [
                {{
                    "stepNumber": 1,
                    "action": "Name of action #1"
                }}
            ],
            "tooling": [
                {{
                    "toolingName": "Name of the tooling",
                    "standard": "ISO-standard for the following tooling if applicable"
                }}
            ]
        }},
        {{
            "stageNumber": 2,
            "stageName": "Name of the stage #2",
            "equipment": "Name and/or model of equipment",
            "steps": [
                {{
                    "stepNumber": 1,
                    "action": "Name of action #1"
                }},
                {{
                    "stepNumber": 2,
                    "action": "Name of action #2"
                }},
               …
            ],
            "tooling": [
                {{
                    "toolingName": "Name of the tooling",
                    "standard": "ISO-standard for the following tooling if applicable"
                }},
                {{
                    "toolingName": "Name of the tooling",
                    "standard": "ISO-standard for the following tooling if applicable"
                }}
            ]
        }},
    …
   ]
}}

Requirements:
(a) The description of each step should be as accurate, concise and consistent with the actual processing as possible.
(b) The equipment used must be specified with respect to the specifics of the operation.
(c) The stages and their steps must cover the full process: roughing, semi-finishing, finishing, quality-inspection and packaging.
(d) The response should be JSON only, with no additional text.
"""

def _format_holes_description(holes: list[dict]) -> str:
    if not holes:
        return "N/A"
    lines: list[str] = []
    for i, h in enumerate(holes, start=1):
        qty = h.get("hole_quantity") if h.get("hole_quantity") is not None else h.get("count")
        radius = h.get("radius") if h.get("radius") is not None else h.get("radius_mm")
        depth = h.get("depth") if h.get("depth") is not None else h.get("depth_mm")
        if qty is None:
            qty = 1
        if radius is None:
            radius = "unknown"
        if depth is None:
            depth = "unknown"
        lines.append(f"Hole {i}: quantity {qty}, radius {radius} mm, depth {depth} mm")
    return "\n".join(lines)


def get_prompt(
    n_dims: int,
    *,
    length: float | str = "unknown",
    width: float | str = "unknown",
    height: float | str = "unknown",
    holes: list[dict] | None = None,
) -> str:
    if n_dims not in (3, 4, 6):
        raise ValueError(f"Unsupported n_dims: {n_dims}. Expected one of: 3, 4, 6.")
    holes = holes or []
    return PROMPT_TEMPLATE.format(
        n_dims=n_dims,
        length=length,
        width=width,
        height=height,
        hole_count=len(holes),
        holes_description=_format_holes_description(holes),
    )

