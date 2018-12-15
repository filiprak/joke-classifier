import asyncio

import pulsar.api as pulsar
from pulsar.async.proxy import command

from classfiers.network import run_network_instance

NUMBER_NETWORK_INSTANCES = 2

network_instances = []

model = None


def run_network_manager(actor, args={}):
    asyncio.ensure_future(run_network_instances())


async def run_network_instances():
    await spawn_network_instances(NUMBER_NETWORK_INSTANCES)
    for aid in network_instances:
        asyncio.ensure_future(pulsar.send(aid, 'run', run_network_instance, args={}))


async def spawn_network_instances(n=1):
    global network_instances

    if len(network_instances) > 0:
        for aid in network_instances:
            pulsar.get_actor().logger.info(aid)
            if aid:
                await pulsar.send(pulsar.get_actor(), 'kill_actor', aid)
        network_instances = []

    for i in range(n):
        aid = 'network_instance' + str(i)
        inst_proxy = await pulsar.spawn(name='network_instance' + str(i), aid=aid)
        network_instances.append(inst_proxy.aid)


@command()
def model_update(request, message):
    global model
    request.actor.logger.info('updating model: ' + str(message))
    model = message
    return 'ok'


@command()
def progress_update(request, message):
    request.actor.logger.info('got progress update: ' + str(message))
    if 'progress' not in request.actor.extra:
        request.actor.extra['progress'] = dict()
    request.actor.extra['progress'][message['aid']] = message['progress']
    return 'ok'


@command()
def get_progress(request):
    if 'progress' not in request.actor.extra:
        return {}
    else:
        return request.actor.extra['progress']
