import asyncio
import logging
import numpy as np

import pulsar.api as pulsar
from pulsar.async.proxy import command

from classifier_models.network_model import create_model
from classifiers.network import run_network_instance
from utils import average_models, serialize_model, update_model, model_size, model_sizeMB, fill_model

NUMBER_NETWORK_INSTANCES = 2

NETWORK_STATE = {
    'model': None,
    'instances': {},

    'input_length': None,
    'output_length': None,
    'activation': None,
}


def run_network_manager(args={}):
    asyncio.ensure_future(run_network_instances(args))


async def run_network_instances(args):
    num_instances = NUMBER_NETWORK_INSTANCES
    activ = 'relu'
    if 'instances' in args:
        num_instances = args['instances']
    if 'activation' in args:
        activ = args['activation']
    await spawn_network_instances(num_instances)

    dpinfo = await pulsar.send('data_provider', 'data_provider_info')

    input_format = 'hot_vector'
    output_format = 'categorical'

    input_length = dpinfo['model_params']['input_length'][input_format]
    output_length = dpinfo['model_params']['output_length'][output_format]

    if NETWORK_STATE['model'] is None:
        NETWORK_STATE['input_length'] = input_length
        NETWORK_STATE['output_length'] = output_length
        NETWORK_STATE['activation'] = activ
        model = create_model(input_length, output_length, activation=activ)

        logging.info(
            'NETWORK SUPER-MODEL INIT (size = {}, X:{}->Y:{})'.format(model_sizeMB(serialize_model(model)),
                                                                      input_length, output_length))

        NETWORK_STATE['model'] = serialize_model(model)

    instance_args = dict(args, **{'model': NETWORK_STATE['model'], 'in_len': input_length,
                                  'out_len': output_length, 'activation': activ,
                                  'input_format': input_format, 'output_format': output_format})

    #logging.warning(instance_args)

    for (aid, inst) in NETWORK_STATE['instances'].items():
        asyncio.ensure_future(
            pulsar.send(aid, 'run', run_network_instance, args=instance_args)
        )


async def spawn_network_instances(n=1):
    for i in range(n):
        aid = 'network_instance' + str(i)
        if aid not in NETWORK_STATE['instances']:
            inst_proxy = await pulsar.spawn(name='network_instance' + str(i), aid=aid)
            NETWORK_STATE['instances'][aid] = {
                'proxy': inst_proxy,
                'aid': inst_proxy.aid,
                'progress': {},
                'progress_timestamp': 0,
            }
        else:
            NETWORK_STATE['instances'][aid]['progress'] = {}
            NETWORK_STATE['instances'][aid]['progress_timestamp'] = 0


def get_prediction(model, X):
    predictions = model.predict([X])
    Y_pred = np.zeros(shape=predictions.shape)
    for prediction, y in zip(predictions, Y_pred):
        y[prediction.argmax()] = 1
    assert (all(sum(x) > 0 for x in Y_pred))
    return Y_pred[0]


@command()
async def run_network_learning(request, args):
    request.actor.logger.info('run_network_learning')
    run_network_manager(args)
    return 'ok'


@command()
def network_model_update(request, received_model):
    request.actor.logger.info('updating model (averaging)')
    NETWORK_STATE['model'] = average_models([NETWORK_STATE['model'], received_model])
    return 'ok'


@command()
def network_progress_update(request, message):
    request.actor.logger.info('got progress update')

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


@command()
async def test_joke(request, joke):
    tokenized = await pulsar.send('data_provider', 'tokenize_joke', joke)

    model = create_model(NETWORK_STATE['input_length'], NETWORK_STATE['output_length'], activation=NETWORK_STATE['activation'])

    if NETWORK_STATE['model'] is not None:
        fill_model(model, NETWORK_STATE['model'])
        pred_Y = get_prediction(model, tokenized)
        categ_idx = pred_Y.tolist().index(1)
        cname = await pulsar.send('data_provider', 'get_categ_name', categ_idx)
        return cname

    return 'Unknown'
