def clamp01(x: float) -> float:
        return 0.0 if x <= 0.0 else 1.0 if x >= 1.0 else x

def distance_factor(distance: float, softness: float = 1.0) -> float:
        """
        Monotone fall-off in [0..1] with distance (metres).

        • softness < 1  → gentler drop, you can perceive well at a few metres.
        • softness = 1  → roughly “1/(1 + d)” (moderate drop).
        • softness > 1  → steep drop (quickly weakens with distance).
        """
        d = max(distance, 0.0)
        return 1.0 / (1.0 + d**softness)