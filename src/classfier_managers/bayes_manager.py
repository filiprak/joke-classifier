import asyncio
import os

import pulsar.api as pulsar
from pulsar.async.proxy import command

from classfiers.bayes import run_bayes_instance

NUMBER_BAYES_INSTANCES = 1


BAYES_STATE = {
    'model': None,
    'instances': {}
}


def run_bayes_manager(args={}):
    asyncio.ensure_future(run_bayes_instances(args))


async def run_bayes_instances(args):
    await spawn_bayes_instances(NUMBER_BAYES_INSTANCES)
    for (aid, inst) in BAYES_STATE['instances'].items():
        asyncio.ensure_future(pulsar.send(aid, 'run', run_bayes_instance, args=args))


async def spawn_bayes_instances(n=1):

    for i in range(len(BAYES_STATE['instances']), len(BAYES_STATE['instances']) + n):
        aid = 'bayes_instance' + str(i)
        inst_proxy = await pulsar.spawn(name='bayes_instance' + str(i), aid=aid)
        BAYES_STATE['instances'][aid] = {
            'proxy': inst_proxy,
            'aid': inst_proxy.aid,
            'progress': 0,
            'progress_timestamp': 0,
        }

    print(BAYES_STATE['instances'])


@command()
async def run_bayes_learning(request, args):
    request.actor.logger.info('run_bayes_learning: ' + str(args))
    run_bayes_manager(args)
    return 'ok'


@command()
def bayes_model_update(request, message):
    request.actor.logger.info('updating model: ' + str(message))
    BAYES_STATE['model'] = message
    return 'ok'


@command()
def bayes_progress_update(request, message):
    request.actor.logger.info('got progress update: ' + str(message))
    request.actor.logger.info('extra (pid = ' + str(os.getpid()) + '): ' + str(BAYES_STATE))

    if message['timestamp'] > BAYES_STATE['instances'][message['aid']]['progress_timestamp']:
        BAYES_STATE['instances'][message['aid']]['progress'] = message['progress']
        BAYES_STATE['instances'][message['aid']]['progress_timestamp'] = message['timestamp']
    return 'ok'


@command()
def get_bayes_progress(request):
    progress = {}
    for aid in BAYES_STATE['instances']:
        progress[aid] = BAYES_STATE['instances'][aid]['progress']
    return progress


@command()
async def kill_bayes_instances(request):
    if len(BAYES_STATE['instances']) > 0:
        for aid, inst in BAYES_STATE['instances'].items():
            res = await pulsar.send('arbiter', 'kill_actor', aid)
            request.actor.logger.info('kill_actor: ' + aid + ' = ' + str(res))
        BAYES_STATE['instances'] = {}

    return 'ok'
