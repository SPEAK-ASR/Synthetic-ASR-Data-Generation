"""Speaking style generator for ASR synthetic data pipeline.

Generates 100+ unique, non-contradictory speaking styles by combining
attributes from tone, pace, manner, and context buckets.
"""

import random

# Each tuple is a compatible pair of tone/emotion adjectives
TONE_PAIRS = [
    ("warm", "welcoming"),
    ("cold", "distant"),
    ("cheerful", "bright"),
    ("somber", "reflective"),
    ("excited", "enthusiastic"),
    ("calm", "serene"),
    ("serious", "authoritative"),
    ("playful", "lighthearted"),
    ("tense", "anxious"),
    ("confident", "assertive"),
    ("gentle", "tender"),
    ("dramatic", "intense"),
    ("dry", "detached"),
    ("melancholic", "wistful"),
    ("upbeat", "energetic"),
    ("hushed", "reverent"),
    ("joyful", "exuberant"),
    ("nostalgic", "sentimental"),
    ("urgent", "compelling"),
    ("bored", "monotone"),
]

PACE_WORDS = [
    "fast", "slow", "measured", "brisk", "leisurely",
    "deliberate", "flowing", "staccato", "unhurried", "clipped",
]

MANNER_WORDS = [
    "conversationally", "formally", "casually", "in a storytelling style",
    "instructionally", "meditatively", "journalistically", "theatrically",
    "intimately", "professionally", "whimsically", "matter-of-factly",
    "poetically", "in a documentary style", "in a broadcast style",
    "in a lecture style", "in a confessional style", "sarcastically",
    "enthusiastically", "gravely",
]

CONTEXT_SUFFIXES = [
    ", as if reading a bedtime story",
    ", as if presenting the evening news",
    ", as if narrating a nature documentary",
    ", as if hosting a podcast",
    ", as if dictating a formal letter",
    ", as if reading poetry aloud",
    ", as if calling an exciting sports play",
    ", as if speaking to a young child",
    ", as if delivering a TED talk",
    ", as if leaving a voicemail",
    ", as if reading an audiobook",
    ", as if announcing a product launch",
    ", as if recounting a ghost story",
    ", as if presenting a cooking show",
    "",  # no context suffix
    "",  # allow no-suffix more often
    "",
]

# ---------------------------------------------------------------------------
# Contradiction rules
# ---------------------------------------------------------------------------
# Each set of words conflicts with another set.  If a generated combination
# contains *any* word from the left set AND *any* word from the right set,
# the combination is invalid.

_CONFLICT_PAIRS: list[tuple[set[str], set[str]]] = [
    (
        {"fast", "brisk", "clipped", "staccato"},
        {"slow", "leisurely", "unhurried", "flowing", "deliberate"},
    ),
    (
        {"slow", "leisurely", "unhurried", "flowing"},
        {"fast", "brisk", "clipped", "staccato", "urgent"},
    ),
    (
        {"calm", "serene", "gentle", "hushed", "meditative", "meditatively"},
        {"tense", "dramatic", "excited", "urgent", "enthusiastic",
         "enthusiastically", "upbeat"},
    ),
    (
        {"cheerful", "playful", "joyful", "upbeat", "warm"},
        {"somber", "melancholic", "cold", "bored", "tense", "dry"},
    ),
    (
        {"formal", "formally", "professional", "professionally",
         "journalistic", "journalistically", "lecture", "in a lecture style"},
        {"casual", "casually", "whimsical", "whimsically",
         "conversational", "conversationally", "sarcastic", "sarcastically"},
    ),
    (
        {"bored", "monotone", "dry"},
        {"excited", "enthusiastic", "enthusiastically", "dramatic",
         "joyful", "upbeat"},
    ),
]


def _has_contradiction(words: set[str]) -> bool:
    """Return True if the word-set triggers any contradiction rule."""
    for left, right in _CONFLICT_PAIRS:
        if words & left and words & right:
            return True
    return False


def _build_style(tone_pair: tuple[str, str], pace: str,
                 manner: str, suffix: str) -> str:
    """Construct a style string from components."""
    base = (f"Read aloud {manner}, {tone_pair[0]} and {tone_pair[1]}, "
            f"at a {pace} pace{suffix}.")
    return base


def generate_unique_styles(n: int = 150) -> list[str]:
    """Generate *n* unique non-contradictory speaking styles.

    Uses rejection sampling: draw random combinations, check all
    contradiction rules, add to a set until *n* unique styles collected.
    Has a safety cap to avoid infinite loops (max 50 000 attempts).
    """
    seen: set[str] = set()
    styles: list[str] = []
    max_attempts = 50_000
    attempts = 0

    while len(styles) < n and attempts < max_attempts:
        attempts += 1
        tone_pair = random.choice(TONE_PAIRS)
        pace = random.choice(PACE_WORDS)
        manner = random.choice(MANNER_WORDS)
        suffix = random.choice(CONTEXT_SUFFIXES)

        # Collect every descriptive word for contradiction checking
        words: set[str] = {
            tone_pair[0], tone_pair[1], pace, manner,
        }
        # Also add the suffix text (lower-cased, stripped) so rules can match
        if suffix:
            words.add(suffix.strip(", ").lower())

        if _has_contradiction(words):
            continue

        style = _build_style(tone_pair, pace, manner, suffix)
        if style not in seen:
            seen.add(style)
            styles.append(style)

    return styles


# Module-level constant
ALL_STYLES: list[str] = generate_unique_styles(150)
