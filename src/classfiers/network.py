import asyncio

import pulsar.api as pulsar


def run_network_instance(actor, args={}):
    asyncio.ensure_future(network_instance_process(actor, args))


async def network_instance_process(actor, args={}):
    for i in range(500001):
        # actor.logger.info(str(i))
        model = i
        for j in range(1000):
            z = 6 * j + j * j
        if i % 10000 == 0:
            await pulsar.send('network_manager_actor', 'progress_update', {'aid': actor.aid, 'progress': i})
            await pulsar.send('network_manager_actor', 'model_update', model)
    return False
