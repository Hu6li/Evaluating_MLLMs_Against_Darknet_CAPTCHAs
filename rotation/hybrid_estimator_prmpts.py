SYSTEM_PROMPT = (
    "You are an expert at analysing circular jigsaw puzzle images. "
    "You have access to a specialised image processing function called "
    "'solve_rotation_polar_gradient' that estimates the clockwise rotation "
    "of the inner disc using computer vision.\n\n"
    "Your workflow MUST be:\n"
    "1. Always call the 'solve_rotation_polar_gradient' function first to "
    "obtain the tool's proposed rotation.\n"
    "2. Critically evaluate whether the proposed rotation is plausible by "
    "visually analysing the image.\n"
    "3. If the tool's proposal looks correct, use it. If it is wrong or "
    "uncertain, correct it based on your own analysis.\n\n"
    "You must always return exactly one rotation value."
)

USER_PROMPT = (
    "The image shows a circular puzzle with two concentric pieces:\n"
    "- The OUTER ring  (fixed, correct orientation)\n"
    "- The INNER disc  (rotated by an unknown angle)\n\n"
    "Task:\n"
    "1. Call the provided function 'solve_rotation_polar_gradient' to get "
    "the tool's suggested rotation.\n"
    "2. Independently verify the result by examining the image.\n"
    "3. Decide on the final clockwise rotation needed to align the inner "
    "disc with the outer ring.\n\n"
    "The value must be an integer between 0 and 359."
)

TOOL_DESCRIPTION = (
    "Classical CV solver that estimates how many degrees CLOCKWISE the inner "
    "disc must be rotated to align with the outer ring "
    "(polar-gradient cross-correlation on the seam). "
    "Returns rotation_cw (0-359 degrees), confidence (float, higher is "
    "better), and ambiguous (bool)."
)

PATH_DESCRIPTION = "Absolute path to the combined puzzle image file."
