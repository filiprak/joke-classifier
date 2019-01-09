import asyncio
import logging
import os

import pulsar.api as pulsar
from pulsar.async.proxy import command

from classifier_models.network_model import create_model
from classifiers.network import run_network_instance
from utils import average_models, serialize_model, update_model, model_size, model_sizeMB

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

    input_format = 'hot_vector'
    output_format = 'categorical'

    input_length = dpinfo['model_params']['input_length'][input_format]
    output_length = dpinfo['model_params']['output_length'][output_format]
    activ = 'relu'

    if NETWORK_STATE['model'] is None:
        model = create_model(input_length, output_length, activation=activ)

        logging.info(
            'NETWORK SUPER-MODEL INIT (size = {}, X:{}->Y:{})'.format(model_sizeMB(serialize_model(model)),
                                                                                  input_length, output_length))

        NETWORK_STATE['model'] = serialize_model(model)

    for (aid, inst) in NETWORK_STATE['instances'].items():
        asyncio.ensure_future(
            pulsar.send(aid, 'run', run_network_instance,
                        args=dict(args, **{'model': NETWORK_STATE['model'], 'in_len': input_length,
                                           'out_len': output_length, 'activation': activ,
                                           'input_format': input_format, 'output_format': output_format})))


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


@command()
async def run_network_learning(request, args):
    request.actor.logger.info('run_network_learning: ' + str(args))
    run_network_manager(args)
    return 'ok'


@command()
def network_model_update(request, received_model):
    request.actor.logger.info('updating model (averaging)')
    NETWORK_STATE['model'] = average_models([NETWORK_STATE['model'], received_model])
    return 'ok'


@command()
def network_progress_update(request, message):
    request.actor.logger.info('got progress update: ' + str(message))

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
