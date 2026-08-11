from __future__ import annotations

from typing import Literal


OutputMode = Literal["text", "parsing"]


TEXT_PROMPT_TEMPLATE = """Analyze the provided image and follow the user instruction.

User instruction:
{instruction}

Identify the focused item by looking for a visible focus ring, outline, glow,
scale change, or coordinated highlight relative to neighboring peer elements.
Do not infer focus from brightness or color alone.

Answer clearly and directly."""


PARSING_PROMPT_TEMPLATE = """Analyze the provided image and follow the user instruction.

User instruction:
{instruction}

Return only valid JSON matching this UI-element schema:
{
  "coordinate_system": "normalized_0_1000",
  "elements": [
    {
      "text": "visible element text",
      "bbox_norm": [x1, y1, x2, y2],
      "focused": false
    }
  ]
}

Requirements:
- Return only elements relevant to the user instruction.
- Use normalized coordinates from 0 to 1000 with bbox_norm ordered as [x1, y1, x2, y2].
- Every element must contain exactly the fields text, bbox_norm, and focused.
- focused must be a JSON boolean.
- Set "focused": true only when an element has a clear navigation-focus cue
  relative to nearby peer elements, such as a focus ring, outline, glow, scale
  change, or coordinated container highlight.
- Do not infer focus from brightness, color, or selected state alone.
- Do not return Markdown or code fences.
- Do not include explanations outside the JSON.
- The fixed output contract above takes precedence over any conflicting formatting request inside the user instruction.
"""


TRANSITION_PROMPT_TEMPLATE = """You are given two consecutive UI screenshots: a BEFORE image and an AFTER image.

Your task is to extract the UI state information from both images.

For each image, return:

1. `elements`

   - List all clearly visible and semantically meaningful UI elements.
   - Include interactive or state-relevant elements such as menu items, buttons, icons, toggles, tabs, selectable items, text labels, and other identifiable UI objects.
   - Include elements regardless of whether they are focused.
   - Do not include decorative backgrounds, borders, separators, shadows, glare, reflections, or other non-semantic visual details.
   - Use short and consistent names.
   - Do not invent elements that are not clearly visible.

2. `focus_path`

   - Determine the focused UI element using the same procedure for each image independently.
   - Inspect all visible interactive elements and compare each candidate with its nearby peer elements.
   - Identify the element that has the strongest visible navigation-focus treatment relative to its peers.
   - Navigation-focus cues include a visible focus ring, outline, glow, border emphasis, scale change, background/container highlight, or another coordinated visual treatment that distinguishes one navigable element from neighboring elements.
   - Do not require an explicit outline or focus ring if another clear relative focus treatment distinguishes the element from its peers.
   - Do not infer focus from semantic importance, expected navigation behavior, prior knowledge, or layout position.
   - Brightness or color alone is not sufficient unless it forms part of a clear coordinated focus treatment relative to neighboring peer elements.
   - First determine the focused leaf element.
   - The final item in `focus_path` must be that visually focused leaf element.
   - Represent the focused element and only its directly visible hierarchical context from parent to child.
   - Include a parent element only when the parent-child relationship is directly supported by visible UI structure.
   - A section title, heading, nearby label, or container name must not automatically be treated as a parent.
   - Do not invent hierarchy merely because an element is visually located below, beside, or inside a labeled section.
   - If only the leaf element can be determined, return only that element, for example: `["Home"]` or `["Sci-fi"]`.
   - Return an empty list `[]` only when no visible interactive element can be distinguished from its peers as focused after explicitly comparing all reasonable candidates.

Important rules:

- Analyze BEFORE and AFTER independently.
- Determine the focused element in each image independently.
- Do not use the focus result from one image to infer the focus result in the other image.
- Do not use or infer any user action.
- Do not assume any predefined menu hierarchy or expected state transition.
- Do not decide whether the transition is correct.
- Do not provide a state name, title, explanation, confidence score, transition description, appeared/disappeared elements, or any fields other than those defined below.
- Keep equivalent element names consistent between BEFORE and AFTER whenever the visual evidence supports it.
- Output valid JSON only.

Required output schema:

{
  "before": {
    "elements": [],
    "focus_path": []
  },
  "after": {
    "elements": [],
    "focus_path": []
  }
}

"""


def compose_prompt(
    instruction: str,
    *,
    output_mode: OutputMode,
) -> str:
    """Compose the controlled model prompt from a user instruction and mode."""
    if not isinstance(instruction, str):
        raise ValueError("instruction must be a non-empty string")
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("instruction must be a non-empty string")
    if output_mode not in ("text", "parsing"):
        raise ValueError("output_mode must be one of: text, parsing")

    template = TEXT_PROMPT_TEMPLATE if output_mode == "text" else PARSING_PROMPT_TEMPLATE
    # Substitute only the controlled placeholder.  The instruction is never
    # interpreted as a format string, so braces in user input remain literal.
    return template.replace("{instruction}", instruction)


def compose_transition_prompt(instruction: str) -> str:
    """Compose the controlled two-image transition prompt."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    # Transition has one fixed extraction contract.  The caller's instruction
    # is accepted for API symmetry but cannot alter that contract.
    return TRANSITION_PROMPT_TEMPLATE
