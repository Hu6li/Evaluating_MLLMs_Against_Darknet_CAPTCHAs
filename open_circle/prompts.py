SYSTEM_PROMPT = "You are an expert at analysing CAPTCHA images to locate incomplete circular rings."

PROMPT_USER = (
    "You are examining a diagram that contains several circular rings drawn on a "
    "coloured background.  Exactly ONE ring is incomplete — it has a visible gap or "
    "opening in its outline.  All other rings are fully closed.\n\n"
    "Your task: find the incomplete ring and report the position of its centre as "
    "fractions of the image dimensions.  "
    "x_frac=0.0 is the left edge, x_frac=1.0 is the right edge.  "
    "y_frac=0.0 is the top edge, y_frac=1.0 is the bottom edge.\n\n"
    "There is always exactly one incomplete ring — you must always return its position.  "
    "If you are uncertain, return your best guess."
)
