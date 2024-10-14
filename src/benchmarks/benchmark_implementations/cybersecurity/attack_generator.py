#!/usr/bin/env python3
import logging
import random
from dataclasses import dataclass
from functools import partial
from itertools import cycle
from operator import itemgetter
from typing import Callable, Iterable, Optional, Protocol, Tuple, TypeVar


class Attack(Protocol):
    """get_scores will always be called after mutate is called"""

    def get_score(self) -> float: ...

    def pre_mutate(self) -> list["Attack"]: ...

    def post_mutate(self) -> list["Attack"]: ...

    def clone(self) -> "Attack": ...


@dataclass
class PhasesGenerator:
    transformations: Iterable[Callable[[list[Attack]], list[Attack]]]  # | "PhasesGenerator"]
    initial_attack_list: Optional[list[Attack]] = None

    def set_initial_attacks(self, attacks: list[Attack]):
        self.initial_attack_list = attacks

    def __iter__(self):
        assert (
            self.initial_attack_list
            or "Initial attacks must be set using set_initial_attacks method!"
        )

        self.state_current_attacks = self.initial_attack_list
        self.state_iterator = iter(self.transformations)
        self.state_counter = 0
        return self

    def __next__(self) -> list[Attack]:
        assert self.state_current_attacks

        logging.debug(f"STAGE {self.state_counter}:\n\n")
        next_transformation_fn = next(self.state_iterator)

        new_attack_list = next_transformation_fn(self.state_current_attacks)
        self.state_current_attacks = new_attack_list
        self.state_counter += 1
        return new_attack_list


# Some generic functional methods
T = TypeVar("T")


def flatten(nested: list[list[T]]) -> list[T]:
    return [y for ys in nested for y in ys]


def generate_branches(branching_factor: int, attacks_in: list[Attack]) -> list[Attack]:
    # Cloning happens here
    for i in range(branching_factor - 1):
        attacks_in.extend([attack.clone() for attack in attacks_in])

    generated_new_attacks = [attack.pre_mutate() for attack in attacks_in]
    flattened = [y for ys in generated_new_attacks for y in ys]
    return flattened


def shuffle(attacks_in: list[Attack]):
    random.shuffle(attacks_in)
    return attacks_in


def keep_first_k(width: int, attacks_in: list[Attack]):
    return attacks_in[:width]


def prune_scores_below_threshold(pruning_threshold: int, attacks_in: list[Attack]):
    def greater_than_threshold(x: Attack) -> bool:
        return x.get_score() >= pruning_threshold

    return list(filter(greater_than_threshold, attacks_in))


def prune_scores(
    stage_configuration: "StageConfiguration", attacks_in: list[Attack]
) -> list[Attack]:
    pruning_threshold = stage_configuration.pruning_threshold
    width = stage_configuration.width

    scores = [attack.get_score() for attack in attacks_in]

    zipped: list[Tuple[Attack, float]] = list(zip(attacks_in, scores))

    # Kick out attacks with score <= 0
    zipped = list(filter(lambda x: x[1] > pruning_threshold, zipped))

    # Sort in descending order according to score and get first k elements
    zipped = list(sorted(zipped, key=lambda x: x[1], reverse=True))  # type: ignore

    get_first = itemgetter(0)
    attacks: list[Attack] = list(map(get_first, zipped))

    # In this case there are no more attacks
    # Passing long the StopIteration exception will break us out of the main iterator
    if len(attacks) == 0:
        raise StopIteration

    return attacks[:width]


@dataclass
class StageConfiguration:
    branching_factor: int = 2
    pruning_threshold: float = 0
    width: int = 2


def single_stage(stage_config: StageConfiguration, attacks_in: list[Attack]) -> list[Attack]:
    """This is the composition of the previous functions to end up with something similar used in TAP during one iteration"""

    branching_factor = stage_config.branching_factor

    # Generate attacks
    assert attacks_in and isinstance(attacks_in, list)

    attacks = generate_branches(branching_factor, attacks_in)
    logging.debug(f"Mutated attacks: \n{attacks}\n")

    # Make sure to stay randomized
    attacks = shuffle(attacks)

    # Prune scores according to topic
    attacks = prune_scores(stage_config, attacks)
    logging.debug(f"After pruning attacks: \n{attacks}\n")

    attacks = flatten([attack.post_mutate() for attack in attacks])

    # The stage is finished, attacks can be returned
    return attacks


def create_tap_generator() -> PhasesGenerator:
    phase_one_config = StageConfiguration(1, -1, 2)
    phase_one = partial(single_stage, phase_one_config)

    phase_two_config = StageConfiguration(1, 0, 2)
    phase_two = partial(single_stage, phase_two_config)

    # two_step_phase = PhasesGenerator([phase_one, phase_two])
    infinite_generator = cycle([phase_one, phase_two])  # cycle(two_step_phase)
    return PhasesGenerator(infinite_generator)
