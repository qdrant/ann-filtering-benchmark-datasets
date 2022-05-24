import os
from functools import partial

from generators.config import DATA_DIR
from generators.random_data.generate import DataGenerator, generate_data

if __name__ == '__main__':
    generator = DataGenerator(vocab_size=1000)

    # --------------------
    #   INT PAYLOAD
    # --------------------

    generate_data(
        generator=generator,
        size=1_000_000,
        dim=100,
        path=os.path.join(DATA_DIR, "random_ints_1m"),
        num_queries=10_000,
        payload_gen=lambda: {
            "a": generator.random_int(100),
            "b": generator.random_int(100)
        },
        condition_gen=partial(generator.random_match_int, rng=100),
        ncpu=4
    )

    generate_data(
        generator=generator,
        size=100_000,
        dim=2048,
        path=os.path.join(DATA_DIR, "random__ints_100k"),
        num_queries=10_000,
        payload_gen=lambda: {
            "a": generator.random_int(100),
            "b": generator.random_int(100)
        },
        condition_gen=partial(generator.random_match_int, rng=100)
    )
