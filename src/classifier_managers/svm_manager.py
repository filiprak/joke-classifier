import asyncio
import os

import pulsar.api as pulsar
from pulsar.async.proxy import command

from classifiers.svm import run_svm_instance

NUMBER_SVM_INSTANCES = 1


SVM_STATE = {
    'model': None,
    'instances': {}
}


def run_svm_manager(args={}):
    asyncio.ensure_future(run_svm_instances(args))


async def run_svm_instances(args):
    await spawn_svm_instances(NUMBER_SVM_INSTANCES)
    for (aid, inst) in SVM_STATE['instances'].items():
        asyncio.ensure_future(pulsar.send(aid, 'run', run_svm_instance, args=args))


async def spawn_svm_instances(n=1):

    for i in range(len(SVM_STATE['instances']), len(SVM_STATE['instances']) + n):
        aid = 'svm_instance' + str(i)
        inst_proxy = await pulsar.spawn(name='svm_instance' + str(i), aid=aid)
        SVM_STATE['instances'][aid] = {
            'proxy': inst_proxy,
            'aid': inst_proxy.aid,
            'progress': 0,
            'progress_timestamp': 0,
        }

    print(SVM_STATE['instances'])


@command()
async def run_svm_learning(request, args):
    request.actor.logger.info('run_svm_learning: ' + str(args))
    run_svm_manager(args)
    return 'ok'


@command()
def svm_model_update(request, message):
    request.actor.logger.info('updating model: ' + str(message))
    SVM_STATE['model'] = message
    return 'ok'


@command()
def svm_progress_update(request, message):
    request.actor.logger.info('got progress update: ' + str(message))
    request.actor.logger.info('extra (pid = ' + str(os.getpid()) + '): ' + str(SVM_STATE))

    if message['timestamp'] > SVM_STATE['instances'][message['aid']]['progress_timestamp']:
        SVM_STATE['instances'][message['aid']]['progress'] = message['progress']
        SVM_STATE['instances'][message['aid']]['progress_timestamp'] = message['timestamp']
    return 'ok'


@command()
def get_svm_progress(request):
    progress = {}
    for aid in SVM_STATE['instances']:
        progress[aid] = SVM_STATE['instances'][aid]['progress']
    return progress


@command()
async def kill_svm_instances(request):
    if len(SVM_STATE['instances']) > 0:
        for aid, inst in SVM_STATE['instances'].items():
            res = await pulsar.send('arbiter', 'kill_actor', aid)
            request.actor.logger.info('kill_actor: ' + aid + ' = ' + str(res))
        SVM_STATE['instances'] = {}

    return 'ok'
