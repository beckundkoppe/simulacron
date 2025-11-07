from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import math
from typing import Dict
import random

from config import PerceptionType
import config
from debug.settings import DEBUG_PERCEPTION_FILTER, DEBUG_PERCEPTION_ENABLED
import util


DEPTH_BLOCK = -1
class Depth(Enum):
    NONE        = -1
    MINIMAL     = 0
    REDUCED     = 1
    BASIC       = 2
    NORMAL      = 3
    EXTENDED    = 4
    RICH        = 5
    FULL        = 6
    REVEAL      = FULL + 1   # drills down into every leaf of the tree but can be blocked by 'VISIBILITY_BLOCK'
    OMNISCIENT  = FULL + 2   # literally all information: drills down into every leaf of the tree and can not be blocked

DEPTH_BLOCK = -1

class Depth(Enum):
    NONE = -1
    MINIMAL = 0
    REDUCED = 1
    BASIC = 2
    NORMAL = 3
    EXTENDED = 4
    RICH = 5
    FULL = 6
    REVEAL = FULL + 1  # drills down into every leaf of the tree but can be blocked by 'VISIBILITY_BLOCK'
    OMNISCIENT = FULL + 2  # literally all information: drills down into every leaf of the tree and can not be blocked

    def reduced(self, visibility: float) -> "Depth":
        """
        Reduce the current depth based on a visibility factor in [0.0, 1.0].
        - visibility == 0.0  → maximal reduction (to MINIMAL, unless blocked/special)
        - visibility == 1.0  → no reduction
        - linear interpolation between these extremes for values in between
        Preserves special logic for OMNISCIENT, REVEAL, and DEPTH_BLOCK.
        """
        if self is Depth.OMNISCIENT:
            return Depth.OMNISCIENT

        if visibility <= 0.0:  # treat negative or zero as full block/reduction
            return Depth.NONE if self.value == DEPTH_BLOCK else Depth.MINIMAL

        if self is Depth.REVEAL:
            return Depth.REVEAL

        if visibility >= 1.0:
            return self  # no reduction

        # Map visibility [0.0, 1.0] → reduction steps [max_steps, 0]
        # We consider the range from MINIMAL to FULL as the reducible part
        max_reducible = Depth.FULL.value - Depth.MINIMAL.value
        reduction_steps = max_reducible * (1.0 - visibility)

        new_value = max(self.value - reduction_steps, Depth.MINIMAL.value)
        # Round down to nearest defined depth level
        stepped = Depth.MINIMAL.value + int(new_value - Depth.MINIMAL.value)
        return Depth(stepped)

    def obfuscate_number(self, n: int) -> str:
        assert self is not Depth.NONE, "NONE should not expose numbers"

        if n == 0:
            return "nothing"

        if self is Depth.MINIMAL:
            # only convey existence vs. non-existence,
            return "some"

        elif self is Depth.REDUCED:
            # very vague quantity terms.
            # Small counts (1–4) → "a few"
            # Larger counts (≥5) → "several"
            # Exact numbers are hidden.
            return "a few" if n < 5 else "several"

        elif self is Depth.BASIC:
            # rough approximation: round to the nearest 10
            # and add a small ±5% random offset,

            rough = round(n, -1)
            delta = max(1, rough // 20)  # ±5% or at least ±1
            noisy = max(0, rough + random.randint(-delta, delta))
            if(n > 5): return f"around {max(1, noisy)}"


        elif self is Depth.NORMAL:
            # "about N" – close to the exact value,
            # but still slightly uncertain (±5% random).
            delta = max(1, n // 20)  # ±5% or at least ±1
            noisy = n + random.randint(-delta, delta)
            if(n > 10): return f"about {max(1, noisy)}"

        elif self is Depth.EXTENDED:
            # range estimate: report a lower and upper bound
            # with a ±10% tolerance.
            tol = max(1, n // 10)
            low = max(0, n - tol)
            high = n + tol
            if(n > 10): return f"between {max(1, low)} and {max(low + 2, high)}"

        elif self is Depth.RICH:
            # "~N" – approximate symbol, gives the number
            # but without claiming exactness.
            if(n > 10): return f"~{n}"

        return str(n)

class SensoryChannel(Enum):
    VISION = auto()
    HEARING = auto()
    SMELL = auto()
    TOUCH = auto()
    TASTE = auto()

class Interaction(Enum):
    # -- no perception ---
    OPEN = auto()
    CLOSE = auto()
    LOCK = auto()
    UNLOCK = auto()

    # --- passive ---
    SURVEY = auto()         # broad, passive intake of the strongest stimuli

    # --- low-effort perception ---
    GLANCE = auto()         # very brief look
    LOOK = auto()           # normal look
    INSPECT = auto()        # finer look

    LISTEN = auto()         # active listening
    SNIFF = auto()          # scent check near object

    # --- medium-effort perception ---
    TOUCH = auto()          # general touch; texture/temperature
    FEEL = auto()           # careful palpation; seams/shapes
    TAP = auto()            # gentle tap/knock; cavity/rattle cues

    # --- high-effort perception ---
    TASTE = auto()
    PUSH = auto()
    PULL = auto()
    SLIDE = auto()
    LIFT = auto()
    TILT = auto()
    SHAKE = auto()
    PEEK = auto()           # quick look through gap/lid

    def channel_effort(self, channel: SensoryChannel) -> float:
        """
        Effort factor for the given channel in a specific interaction.
        Defaults to 1.0 if that channel is not part of the interaction.
        """
        for ic in INTERACTION_SENSES.get(self, set()):
            if ic.channel is channel:
                return ic.effort
        return 1.0

@dataclass(frozen=True)
class InteractionSense:
    channel: SensoryChannel
    effort: float = 1.0         # (0.0=none, 1.0=normal)

INTERACTION_SENSES: Dict[Interaction, InteractionSense] = {
    Interaction.SURVEY:  {
        InteractionSense(SensoryChannel.VISION,   0.9),
        InteractionSense(SensoryChannel.HEARING,  0.4),
        InteractionSense(SensoryChannel.SMELL,    0.2),
        InteractionSense(SensoryChannel.TOUCH,    0.2),
        InteractionSense(SensoryChannel.TASTE,    0.1)
    },

    Interaction.GLANCE:  {InteractionSense(SensoryChannel.VISION)},
    Interaction.LOOK:    {InteractionSense(SensoryChannel.VISION)},
    Interaction.INSPECT: {InteractionSense(SensoryChannel.VISION)},
    Interaction.LISTEN:  {InteractionSense(SensoryChannel.HEARING)},
    Interaction.SNIFF:   {InteractionSense(SensoryChannel.SMELL)},

    Interaction.TOUCH:   {InteractionSense(SensoryChannel.TOUCH)},
    Interaction.FEEL:    {InteractionSense(SensoryChannel.TOUCH)},
    Interaction.TAP:     {InteractionSense(SensoryChannel.TOUCH), InteractionSense(SensoryChannel.HEARING)},

    Interaction.PUSH:    {InteractionSense(SensoryChannel.TOUCH)},
    Interaction.PULL:    {InteractionSense(SensoryChannel.TOUCH)},
    Interaction.SLIDE:   {InteractionSense(SensoryChannel.TOUCH)},
    Interaction.LIFT:    {InteractionSense(SensoryChannel.TOUCH)},
    Interaction.TILT:    {InteractionSense(SensoryChannel.TOUCH)},
    Interaction.SHAKE:   {InteractionSense(SensoryChannel.TOUCH), InteractionSense(SensoryChannel.HEARING)},

    Interaction.OPEN:    {InteractionSense(SensoryChannel.TOUCH), InteractionSense(SensoryChannel.VISION), InteractionSense(SensoryChannel.HEARING)},
    Interaction.CLOSE:   {InteractionSense(SensoryChannel.TOUCH), InteractionSense(SensoryChannel.VISION), InteractionSense(SensoryChannel.HEARING)},
    Interaction.PEEK:    {InteractionSense(SensoryChannel.VISION)},

    Interaction.TASTE:   {InteractionSense(SensoryChannel.TASTE, 0.8)},
}

@dataclass(frozen=True)
class InteractionMeta:
    cost: float = 1.0                   # (0.0=none, 1.0=normal)
    risk_exposure: float = 0.01         # (0.0=none, 1.0=high)
    noise_generated: float = 1.0        # (0.0=none, 1.0=normal)
    time_required_s: float = 1.0

INTERACTION_METAS: dict[Interaction, InteractionMeta] = {
    Interaction.SURVEY: InteractionMeta(
        cost   = 0.0,
        noise_generated = 0.1
    ),

    # --- low-effort perception ---
    Interaction.GLANCE: InteractionMeta(time_required_s=0.1),
    Interaction.LOOK:   InteractionMeta(),
    Interaction.INSPECT:InteractionMeta(time_required_s=3.0, cost=0.2),
    Interaction.LISTEN: InteractionMeta(time_required_s=2.0),
    Interaction.SNIFF:  InteractionMeta(time_required_s=1.5),

    # --- light touch / probing without tools ---
    Interaction.TOUCH:  InteractionMeta(),
    Interaction.FEEL:   InteractionMeta(time_required_s=2.5, cost=0.5, risk_exposure=0.2),
    Interaction.TAP:    InteractionMeta( noise_generated=0.4),

    # --- gentle manipulation ---
    Interaction.PUSH:   InteractionMeta(cost=0.3, noise_generated=0.2),
    Interaction.PULL:   InteractionMeta(cost=0.3, noise_generated=0.2),
    Interaction.SLIDE:  InteractionMeta(time_required_s=1.5, cost=0.4, noise_generated=0.3),
    Interaction.LIFT:   InteractionMeta(time_required_s=2.0, cost=0.8, noise_generated=0.3),
    Interaction.TILT:   InteractionMeta(time_required_s=2.0, cost=0.6, noise_generated=0.3),
    Interaction.SHAKE:  InteractionMeta(time_required_s=2.0, cost=0.8, noise_generated=0.7, risk_exposure=0.1),

    # --- container state ---
    Interaction.OPEN:   InteractionMeta(time_required_s=2.0, cost=0.5, noise_generated=0.2),
    Interaction.CLOSE:  InteractionMeta(cost=0.3, noise_generated=0.2),
    Interaction.PEEK:   InteractionMeta(),

    # --- risky ---
    Interaction.TASTE:  InteractionMeta(time_required_s=1.5, risk_exposure=1.0),
}

@dataclass
class PerceptionEnviroment:
    """
    Perception context.

    distance_m : float
        >=0, distance to target (affects all sense falloffs).
    light_level : float
        0..1, 0=dark, 1=very bright daylight. Default 0.5.
    ambient_noise : float
        0..1, 0=silent, 1=very loud. Default 0.5.
    ambient_smell : float
        0..1, 0=neutral air, 1=strong background odor. Default 0.5.
    interactions : Set[Interaction]
        Active purely physical interactions.
    """
    distance_m: float
    light_level: float = 0.5
    ambient_noise: float = 0.5
    ambient_smell: float = 0.5
    interactions: list[Interaction] = field(default_factory=set)
    
    def vision_effective(self) -> float:
        """
        Visual perception.
        Exponential falloff with distance, moderated by light.
        ~0.8 at 5 m daylight, ~0.6 at 10 m.
        """
        if self.distance_m > 60:
            return 0.0
        return _channel_effective(
            self.distance_m,
            softness=0.05,               # slow, realistic optical falloff
            base=min(1.0, self.light_level)  # >1.0 brightness gives no extra benefit
        )

    def hearing_effective(self) -> float:
        """
        Auditory perception.
        Faster distance decay than vision, reduced by ambient noise.
        ~normal speech range fades near 20 m.
        """
        if self.distance_m > 20:
            return 0.0
        noise_mask = max(0.0, 1.0 - self.ambient_noise)
        return _channel_effective(
            self.distance_m,
            softness=0.6,      # steeper falloff for sound
            mask=noise_mask
        )

    def smell_effective(self) -> float:
        """
        Olfactory perception.
        Similar to hearing in distance reach,
        reduced by background odors.
        """
        if self.distance_m > 15:
            return 0.0
        smell_mask = max(0.0, 1.0 - self.ambient_smell)
        return _channel_effective(
            self.distance_m,
            softness=0.9,      # strong attenuation with distance
            mask=smell_mask
        )

    def touch_effective(self) -> float:
        """
        Tactile perception.
        Essentially local: negligible beyond 0.5 m.
        """
        if self.distance_m > 0.5:
            return 0.0
        return _channel_effective(
            self.distance_m,
            softness=0.5      # steep, but allows a gentle roll-off near 0.5 m
        )

    def signal_strength(self, channel: SensoryChannel) -> float:
        """
        Return the effective 0..1 signal strength for the requested sense.
        Combines distance and ambient conditions using the same logic
        as the individual *_effective() helpers.
        """
        if not self.allows(channel):
            return 0.0

        if channel is SensoryChannel.VISION:
            return self.vision_effective()
        if channel is SensoryChannel.HEARING:
            return self.hearing_effective()
        if channel is SensoryChannel.SMELL:
            return self.smell_effective()
        if channel is SensoryChannel.TOUCH:
            return self.touch_effective()
        if channel is SensoryChannel.TASTE:
            # taste requires direct contact; assume full signal if allowed
            return 1.0
        return 0.0
        
    def active_channels(self) -> set[SensoryChannel]:
        """
        Return the set of pure SensoryChannel enums
        that the current interactions make available.
        """
        chans: set[SensoryChannel] = set()
        for action in self.interactions:
            for ic in INTERACTION_SENSES.get(action, set()):
                chans.add(ic.channel)     # Zugriff auf das Enum-Attribut
        return chans

    def allows(self, channel: SensoryChannel) -> bool:
        """
        True if the current interactions allow perception on the
        given sensory channel.
        """
        return channel in self.active_channels()
    
    def can_use_vision(self) -> bool:
        """True if vision-based perception is currently possible."""
        return self.allows(SensoryChannel.VISION)

    def can_use_hearing(self) -> bool:
        """True if hearing-based perception is currently possible."""
        return self.allows(SensoryChannel.HEARING)

    def can_use_smell(self) -> bool:
        """True if smell-based perception is currently possible."""
        return self.allows(SensoryChannel.SMELL)

    def can_use_touch(self) -> bool:
        """True if touch-based perception is currently possible."""
        return self.allows(SensoryChannel.TOUCH)

    def can_use_taste(self) -> bool:
        """True if taste-based perception is currently possible."""
        return self.allows(SensoryChannel.TASTE)

def _channel_effective(
        distance_m: float,
        softness: float = 0.5,
        base: float = 1.0,
        mask: float = 1.0,
        max_effect: float = 1.0,
        offset: float = 0.0,
        kind: str = "exp"
    ) -> float:
        """
        Generic falloff calculator for perception channels.

        Parameters
        ----------
        distance_m : float
            Distance to the source in meters.
        softness : float
            Controls steepness of falloff:
            - larger = steeper decay (for "exp": faster exponent).
        base : float
            Base signal strength (e.g. brightness) 0..1.
        mask : float
            Multiplicative reduction due to ambient interference (0..1).
        max_effect : float
            Maximum allowed effective value (default 1.0).
        offset : float
            Constant added to the falloff before clamping.
        kind : {"exp","linear","quad"}
            Shape of the distance falloff.

        Returns
        -------
        float
            Effective signal in [0..max_effect].
        """
        if distance_m <= 0:
            falloff = 1.0
        elif kind == "linear":
            falloff = max(0.0, 1.0 - distance_m * softness)
        elif kind == "quad":
            falloff = max(0.0, 1.0 - (distance_m * softness) ** 2)
        else:  # "exp" by default
            falloff = math.exp(-distance_m * softness)

        raw = (base * mask) * (falloff + offset)
        return min(max_effect, util.clamp01(raw))

@dataclass
class ObserverPerception:
    """
    Individual perception modifiers – describes the observer.
    """
    def __init__(
        self,
        hearing_sensitivity_db: float = 20.0,  # dB SPL threshold for normal hearing
        vision_acuity: float = 1.0,            # 1.0 = normal eyesight
        smell_sensitivity: float = 1.0,        # 1.0 = normal sense of smell
        fatigue: float = 0.0,                  # 0..1 (1 = very tired)
        attention: float = 1.0                 # 0..1 (1 = fully attentive)
    ) -> None:
        self.hearing_sensitivity_db = hearing_sensitivity_db
        self.vision_acuity = vision_acuity
        self.smell_sensitivity = smell_sensitivity
        self.fatigue = fatigue
        self.attention = attention

    # ----------------------------------------------------------------------
    # HEARING
    # ----------------------------------------------------------------------
    def _hearing_physical(self,
                          env: PerceptionEnviroment,
                          source_level_db: float = 60.0,
                          dynamic_range_db: float = 40.0) -> float:
        """
        Raw audibility 0..1 of a sound source given environmental and personal factors.
        Uses env.hearing_effective() as the base environmental signal.
        """
        # base signal purely from environment
        env_signal = env.hearing_effective()

        # approximate conversion of dB SPL to 0..1 loudness
        if env.distance_m <= 0:
            level = source_level_db
        else:
            level = source_level_db - 20 * math.log10(env.distance_m)

        margin = max(0.0, level - self.hearing_sensitivity_db)
        personal_factor = min(1.0, margin / dynamic_range_db)

        # personal state: attention vs. fatigue
        personal_mask = max(0.0, self.attention * (1.0 - self.fatigue))

        return env_signal * personal_factor * personal_mask


    def understand_speech_effective(self, env: PerceptionEnviroment, source_level_db: float = 60.0) -> float:
        """
        Probability (0..1) of understanding normal speech
        (~60 dB @ 1 m) under current conditions.
        """
        raw = self._hearing_physical(env, source_level_db)
        # need a better SNR than mere audibility
        return max(0.0, raw - 0.2)  # simple extra penalty for intelligibility

    def hear_sound_effective(self, env: PerceptionEnviroment, source_level_db: float = 80.0) -> float:
        """
        Probability (0..1) of detecting a loud alarm or mechanical wecker
        (~80 dB @ 1 m).
        """
        return self._hearing_physical(env, source_level_db)
    
    # ----------------------------------------------------------------------
    # VISION
    # ----------------------------------------------------------------------
    def vision_effective(self, env: PerceptionEnviroment, intensity: float = 1.0) -> float:
        """
        Effective visual perception.
        Combines the environmental base from env.vision_effective()
        with personal modifiers such as acuity and fatigue.
        """
        base = env.vision_effective() * intensity
        personal_mask = max(0.0, self.vision_acuity * (1.0 - self.fatigue))
        return max(0.0, min(1.0, base * personal_mask))

    # ----------------------------------------------------------------------
    # SMELL
    # ----------------------------------------------------------------------
    def smell_effective(self, env: PerceptionEnviroment, intensity: float = 1.0) -> float:
        """
        Effective olfactory perception.
        Uses env.smell_effective() as base and applies personal sensitivity.
        """
        base = env.smell_effective() * intensity
        personal_mask = max(0.0, self.smell_sensitivity * (1.0 - self.fatigue))
        return max(0.0, min(1.0, base * personal_mask))

    # ----------------------------------------------------------------------
    # TOUCH
    # ----------------------------------------------------------------------
    def touch_effective(self, env: PerceptionEnviroment, intensity: float = 1.0) -> float:
        """
        Effective tactile perception.
        Uses env.touch_effective() as base and applies fatigue penalty.
        """
        base = env.touch_effective() * intensity
        # assume normal touch sensitivity = 1.0; can be extended if you add attribute
        personal_mask = max(0.0, (1.0 - self.fatigue))
        return max(0.0, min(1.0, base * personal_mask))
    
class DatumOperator(Enum):
    OR     = auto()
    AND    = auto()
    
class Datum:
    def __init__(
        self,
        entity: Entity,
        key: str,
        value: str,
        channel_intensity: Dict[SensoryChannel, float],
        min_depth: Depth = Depth.BASIC,         # narrative / logical requirement
        detect_threshold: float = 0.5,          # physical signal strength 0..1
        op: DatumOperator = DatumOperator.OR,   # by default one channel is enought
    ) -> None:
        self.entity = entity
        self.key = key
        self.value = value
        self.channel_requirements = channel_intensity
        self.min_detection_level = min_depth
        self.detect_threshold = detect_threshold
        self.op = op

    def perceive(
        self,
        observer: "ObserverPerception",
        env: "PerceptionEnviroment",
        level: Depth,
    ) -> None:
        discovery = config.CONFIG.perception

        if (
            (discovery == PerceptionType.DISTANCE and self.is_perceived_simple(observer, env, level))
            or (discovery == PerceptionType.SENSE and self.is_perceived(observer, env, level))
            or (discovery == PerceptionType.FULL)
        ):
            self.entity.info[self.key] = self.value
        else:
            self.entity.info["object"] = "unknown"

    def is_perceived_simple(
        self,
        observer: "ObserverPerception",
        env: "PerceptionEnviroment",
        level: Depth,
    ) -> bool:
        if env.distance_m <= config.DISTANCE:
            return True
        else:
            return False

    def is_perceived(
        self,
        observer: "ObserverPerception",
        env: "PerceptionEnviroment",
        level: Depth,
    ) -> bool:
        """
        Return True if this datum is perceived by `observer` in `env`.

        Uses self.op (DatumOperator.AND / OR) to combine the
        results of all required sensory channels.
        """

        # --- Boost depending on requested depth
        boost = 1 + (level.value / Depth.FULL.value) * 0.4

        is_debug_mode = self._match_debug_filter(observer)
        if is_debug_mode:
            debug_bar = "=" * 72
            debug_sep = "-" * 72
            print(f"\n{debug_bar}")
            print("[ PERCEPTION DEBUG ]\n")
            print(f"Observer : {observer.name}")
            print(f"Entity   : {self.entity.readable_id} ({self.entity.uuid})")
            print(f"Datum    : {self.key}")
            print(f"Required Depth : {self.min_detection_level.name}")
            print(f"Operator       : {self.op.name}")
            print(f"Boost from requested level ({level.name}) : {boost:.2f}")

        # --- quick exit if requested depth is below the datum's minimum
        if level.value < self.min_detection_level.value:
            if is_debug_mode:
                print("→ Requested depth below minimum")
                print(debug_bar)
            return False

        active = env.active_channels()
        if is_debug_mode:
            print(f"Active channels  : {', '.join(ch.name for ch in active)}")
            print(f"Required channels: {', '.join(ch.name for ch in self.channel_requirements)}")

        # --- quick exit if none of the required channels is active
        if not any(ch in active for ch in self.channel_requirements):
            if is_debug_mode:
                print("→ No required sensory channel is active")
                print(debug_bar)
            return False

        if is_debug_mode:
            print(debug_sep)
            print("{:<8} {:>5} {:>7} {:>7} {:>10} {:>8} {:>10}".format(
                "Channel", "Src", "Score", "Effort",
                "AfterEff", "Boost", "Final"
            ))
            print(debug_sep)

        # collect per-channel results for AND/OR logic
        channel_results: list[bool] = []

        for ch_enum, intensity in self.channel_requirements.items():
            if ch_enum not in active:
                # required channel not active -> fails for AND,
                # ignore for OR
                if self.op == DatumOperator.AND:
                    channel_results.append(False)
                continue

            intensity = intensity * self.entity.prominence

            # --- base perception score
            if ch_enum is SensoryChannel.VISION:
                score = observer.vision_effective(env, intensity=intensity)
            elif ch_enum is SensoryChannel.HEARING:
                score = observer.hear_sound_effective(env, source_level_db=intensity)
            elif ch_enum is SensoryChannel.SMELL:
                score = observer.smell_effective(env, intensity=intensity)
            elif ch_enum is SensoryChannel.TOUCH:
                score = observer.touch_effective(env, intensity=intensity)
            elif ch_enum is SensoryChannel.TASTE:
                score = intensity
            else:
                continue

            # --- combine efforts with diminishing bonus
            efforts: list[float] = []
            for action in env.interactions:
                for ic in INTERACTION_SENSES.get(action, set()):
                    if ic.channel is ch_enum:
                        efforts.append(ic.effort)

            if not efforts:
                combined_effort = 1.0
            else:
                efforts.sort(reverse=True)
                combined_effort = efforts[0]
                for e in efforts[1:]:
                    combined_effort += 0.1 * e
                combined_effort = min(combined_effort, 1.5)

            score_after_effort = min(1.0, score * combined_effort)
            final_score = min(1.0, score_after_effort * boost)
            passed = final_score >= self.detect_threshold
            channel_results.append(passed)

            if is_debug_mode:
                print("{:<8} {:>5.2f} {:>7.2f} {:>7.2f} {:>10.2f} {:>8.2f} {:>10.2f} {}".format(
                    ch_enum.name,
                    intensity,
                    score,
                    combined_effort,
                    score_after_effort,
                    boost,
                    final_score,
                    "✓" if passed else "✗"
                ))

        # --- combine results according to the operator
        if self.op == DatumOperator.AND:
            result = all(channel_results)
        else:  # OR (default)
            result = any(channel_results)

        if is_debug_mode:
            print(debug_sep)
            print(f"Threshold : {self.detect_threshold:.2f}")
            if result:
                print("→ Overall threshold met")
                print(f"   └─ perceived value: \"{self.value}\"")
            else:
                print("→ Overall threshold NOT met")
            print(debug_bar)

        return result

    
    def _match_debug_filter(self, observer: ObserverPerception) -> bool:
            if DEBUG_PERCEPTION_ENABLED is False:
                return False

            if not DEBUG_PERCEPTION_FILTER:
                return True

            for e_name, o_name, d_key in DEBUG_PERCEPTION_FILTER:
                if ((e_name == "*" or e_name == self.entity.readable_id)
                    and (o_name == "*" or o_name == observer.name)
                    and (d_key   == "*" or d_key   == self.key)):
                    return True
            return False
    
class CompositeDatum(Datum):
    """
    A datum that evaluates a custom boolean condition
    over sensory scores.
    """
    def __init__(
            self,
            entity: Entity,
            key: str,
            value: str,
            condition,
            min_depth: Depth = Depth.BASIC,         # narrative / logical requirement
            ):
        super().__init__(
            entity,
            key=key,
            value=value,
            channel_intensity={},  # not used directly
            min_depth=min_depth,
            detect_threshold=0.0,      # not used directly
        )
        self.condition = condition   # callable() -> bool

    def is_perceived(self, observer, env, level: Depth) -> bool:
        if level.value < self.min_detection_level.value:
            return False
        return self.condition()
