import os

from generators.config import DATA_DIR
from generators.generate import DataGenerator, generate_random_dataset

if __name__ == '__main__':
    generator = DataGenerator(vocab_size=1000)

    # --------------------
    #   FLOAT PAYLOAD
    # --------------------

    generate_random_dataset(
        generator=generator,
        size=1_000_000,
        dim=100,
        path=os.path.join(DATA_DIR, "random_float_1m"),
        num_queries=10_000,
        payload_gen=lambda: {
            "a": generator.random_float(),
            "b": generator.random_float()
        },
        condition_gen=generator.random_range_query,
    )

    generate_random_dataset(
        generator=generator,
        size=100_000,
        dim=2048,
        path=os.path.join(DATA_DIR, "random__float_100k"),
        num_queries=10_000,
        payload_gen=lambda: {
            "a": generator.random_float(),
            "b": generator.random_float()
        },
        condition_gen=generator.random_range_query,
    )
