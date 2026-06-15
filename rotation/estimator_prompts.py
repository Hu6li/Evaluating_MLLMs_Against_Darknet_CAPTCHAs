PROMPT_SYSTEM = """You are an expert at analysing circular jigsaw puzzle images."""

PROMPT_USER = """\
The image shows a circular puzzle with two concentric pieces:
- The OUTER ring  (fixed, correct orientation)
- The INNER disc  (rotated by an unknown angle)

The visual pattern on the inner disc should align seamlessly with the outer ring \
when the inner disc is rotated to the correct position.

Your task: estimate how many degrees CLOCKWISE the inner disc must be rotated \
to align it with the outer ring.

The value must be an integer between 0 and 359.
"""
