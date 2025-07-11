import mmh3


def to_color(name: str) -> str:
    """Convert name to a color using the murmur3 hash"""

    # Somewhat vibrant color palette.
    # These names must match the -square and -chip classes in input.css.
    colors = [
        'coral',
        'turquoise',
        'sky-blue',
        'gray-aquamarine',
        'pale-yellow',
        'dusky-rose',
        'purple',
        'medium-pink',
        'bright-blue',
        'aquamarine',
    ]

    hash = abs(mmh3.hash(name))
    return colors[hash % len(colors)]
