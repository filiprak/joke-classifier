import asyncio
import os

import pulsar.api as pulsar
import tflearn
from pulsar.async.proxy import command

from classfiers.network import run_network_instance

NUMBER_NETWORK_INSTANCES = 3


NETWORK_STATE = {
    'model': None,
    'instances': {}
}


def run_network_manager(args={}):
    asyncio.ensure_future(run_network_instances(args))


async def run_network_instances(args):
    await spawn_network_instances(NUMBER_NETWORK_INSTANCES)

    dpinfo = await pulsar.send('data_provider', 'data_provider_info')

    model = create_model(100, 100)

    NETWORK_STATE['model'] = model

    for (aid, inst) in NETWORK_STATE['instances'].items():
        asyncio.ensure_future(pulsar.send(aid, 'run', run_network_instance, args=dict(args, **{'model': model})))


async def spawn_network_instances(n=1):

    for i in range(n):
        aid = 'network_instance' + str(i)
        if aid not in NETWORK_STATE['instances']:
            inst_proxy = await pulsar.spawn(name='network_instance' + str(i), aid=aid)
            NETWORK_STATE['instances'][aid] = {
                'proxy': inst_proxy,
                'aid': inst_proxy.aid,
                'progress': 0,
                'progress_timestamp': 0,
            }

    print(NETWORK_STATE['instances'])


def create_model(input_length, output_length, activation='relu'):
    input_layer = tflearn.input_data(shape=[None, input_length])
    model = tflearn.fully_connected(input_layer,
                                    64,
                                    activation=activation)
    model = tflearn.dropout(model, 0.8)
    model = tflearn.fully_connected(input_layer,
                                    64,
                                    activation=activation)
    model = tflearn.dropout(model, 0.8)
    softmax = tflearn.fully_connected(model, output_length, activation='softmax')
    sgd = tflearn.SGD(learning_rate=0.1, decay_step=1000)
    net = tflearn.regression(softmax,
                             optimizer=sgd,
                             loss='categorical_crossentropy')
    model = tflearn.DNN(net, tensorboard_verbose=0)
    return model


@command()
async def run_network_learning(request, args):
    request.actor.logger.info('run_network_learning: ' + str(args))
    run_network_manager(args)
    return 'ok'


@command()
def network_model_update(request, message):
    request.actor.logger.info('updating model: ' + str(message))
    NETWORK_STATE['model'] = message
    return 'ok'


@command()
def network_progress_update(request, message):
    request.actor.logger.info('got progress update: ' + str(message))
    request.actor.logger.info('extra (pid = ' + str(os.getpid()) + '): ' + str(NETWORK_STATE))

    if message['timestamp'] > NETWORK_STATE['instances'][message['aid']]['progress_timestamp']:
        NETWORK_STATE['instances'][message['aid']]['progress'] = message['progress']
        NETWORK_STATE['instances'][message['aid']]['progress_timestamp'] = message['timestamp']
    return 'ok'


@command()
def get_network_progress(request):
    progress = {}
    for aid in NETWORK_STATE['instances']:
        progress[aid] = NETWORK_STATE['instances'][aid]['progress']
    return progress


@command()
async def kill_network_instances(request):
    if len(NETWORK_STATE['instances']) > 0:
        for aid, inst in NETWORK_STATE['instances'].items():
            res = await pulsar.send('arbiter', 'kill_actor', aid)
            request.actor.logger.info('kill_actor: ' + aid + ' = ' + str(res))
        NETWORK_STATE['instances'] = {}

    return 'ok'


