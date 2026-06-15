SYSTEM_PROMPT = "You are a biologist reviewing a grid of nature photographs for an insect survey."

PROMPT_USER = """\
Each photograph shows either an insect or a non-insect subject (plants, landscapes, objects, etc.).
Your task is to identify which grid squares contain insects \
(such as flies, mosquitoes, bees, beetles, ants, butterflies, etc.).

The grid is numbered from left to right, top to bottom, starting at 1.

Carefully examine each square and identify ALL squares that contain any type of insect.

If no insects are found, return an empty list with the reason in reasoning.
"""
