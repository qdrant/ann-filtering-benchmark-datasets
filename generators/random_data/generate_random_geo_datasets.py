import os
from functools import partial

from generators.config import DATA_DIR
from generators.generate import DataGenerator, generate_random_dataset

if __name__ == '__main__':
    generator = DataGenerator(vocab_size=1000)

    # --------------------
    #   GEO PAYLOAD
    # --------------------

    generate_random_dataset(
        generator=generator,
        size=1_000_000,
        dim=100,
        path=os.path.join(DATA_DIR, "random_geo_1m"),
        num_queries=10_000,
        payload_gen=lambda: {
            "a": generator.random_geo(),
            "b": generator.random_geo()
        },
        condition_gen=partial(generator.random_geo_query, radius=2_000_000),
    )

    generate_random_dataset(
        generator=generator,
        size=100_000,
        dim=2048,
        path=os.path.join(DATA_DIR, "random_geo_100k"),
        num_queries=10_000,
        payload_gen=lambda: {
            "a": generator.random_geo(),
            "b": generator.random_geo()
        },
        condition_gen=partial(generator.random_geo_query, radius=2_000_000),
    )
