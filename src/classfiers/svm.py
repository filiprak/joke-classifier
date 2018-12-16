import asyncio

import pulsar.api as pulsar
import time

def run_svm_instance(actor, args={}):
    asyncio.ensure_future(svm_instance_process(actor, args))


async def svm_instance_process(actor, args={}):
    for i in range(200001):
        # actor.logger.info(str(i))
        model = i
        for j in range(1000):
            z = 6 * j + j * j
        if i % 10000 == 0:
            await pulsar.send('svm_manager_actor', 'svm_progress_update', {'aid': actor.aid,
                                                                           'timestamp': time.time(),
                                                                           'progress': 100 * i / 200001})
            await pulsar.send('svm_manager_actor', 'svm_model_update', model)
    await pulsar.send('svm_manager_actor', 'svm_progress_update', {'aid': actor.aid,
                                                                   'timestamp': time.time(),
                                                                   'progress': 101})
    return False
