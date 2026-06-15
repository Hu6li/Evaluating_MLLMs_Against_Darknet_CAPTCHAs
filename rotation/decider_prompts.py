system_prompt = """You are an expert at analyzing composite circular images where an inner disc is rotated relative to a fixed outer ring. You excel at detecting misalignments across a circular boundary."""

user_message = """
You are an expert in visual alignment and puzzle solving. I am providing 8 circular images. Each image consists of two concentric parts:

- The OUTER ring is FIXED and in the correct orientation in ALL images.
- The INNER disc is a separate piece that has been rotated (by multiples of 45° in most cases, or by only a few degrees in exactly one case).

Task: Identify which SINGLE image has the INNER disc aligned best with the OUTER ring. The correct alignment should show a nearly seamless visual continuity across the circular boundary.

Important details:
- Exactly ONE image is only a few degrees off from perfect alignment (look for very subtle misalignment, such as tiny discontinuity at the boundary of the inner and outer part).
- All other images are rotated by 45° or more, causing obvious large breaks in the pattern (plaid lines not continuing, colors are not matching).

Analyze the images carefully, focusing on edge continuity across the circular boundary. Do not guess — base your decision strictly on visual pattern matching.

Output format (strictly follow this):
- Correct image name: The exact original filename of the best-aligned image (e.g., "images1_pair0_combined_y.jpg")
- Image number in upload sequence: [X] (e.g., 2)
- Reasoning: Step-by-step explanation of why this one aligns best and why the others do not. Mention specific visual cues (plaid continuity).
- Confidence: XX% (how sure you are, considering possible tiny rotations)

Now analyze the 8 attached images and provide your answer.
"""
