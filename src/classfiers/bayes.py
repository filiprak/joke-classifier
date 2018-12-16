import asyncio
import time

import pulsar.api as pulsar


def run_bayes_instance(actor, args={}):
    asyncio.ensure_future(bayes_instance_process(actor, args))


async def bayes_instance_process(actor, args={}):
    for i in range(200001):
        # actor.logger.info(str(i))
        model = i
        for j in range(1000):
            z = 6 * j + j * j
        if i % 10000 == 0:
            await pulsar.send('bayes_manager_actor', 'bayes_progress_update', {'aid': actor.aid,
                                                                               'timestamp': time.time(),
                                                                                'progress': 100 * i / 200001})
            await pulsar.send('bayes_manager_actor', 'bayes_model_update', model)
    await pulsar.send('bayes_manager_actor', 'bayes_progress_update', {'aid': actor.aid,
                                                                       'timestamp': time.time(),
                                                                        'progress': 101})
    return False
