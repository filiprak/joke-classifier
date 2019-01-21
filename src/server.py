import asyncio
import json
import logging

import pulsar.api as pulsar

import pulsar.apps.wsgi as wsgi
from pulsar.apps.wsgi import AccessControl

import classifier_managers.network_manager, classifier_managers.bayes_manager, classifier_managers.svm_manager
from data_provider import init_data_provider

app = wsgi.Router('/')

arbiter = pulsar.arbiter()
algorithm_opts = ['svm', 'bayes', 'network', 'all']

svm_manager_actor, bayes_manager_actor, network_manager_actor = None, None, None
data_provider = None


async def spawn_managers(algo='all'):
    global arbiter, svm_manager_actor, bayes_manager_actor, network_manager_actor

    if not svm_manager_actor and algo in ['svm', 'all']:
        svm_manager_actor = await pulsar.spawn(name='svm_manager_actor', aid='svm_manager_actor')

    if not bayes_manager_actor and algo in ['bayes', 'all']:
        bayes_manager_actor = await pulsar.spawn(name='bayes_manager_actor', aid='bayes_manager_actor')

    if not network_manager_actor and algo in ['network', 'all']:
        network_manager_actor = await pulsar.spawn(name='network_manager_actor', aid='network_manager_actor')


def init_data_provider_task(actor):
    init_data_provider(ngrams=False)


async def spawn_data_provider():
    global data_provider
    if not data_provider:
        data_provider = await pulsar.spawn(name='data_provider', aid='data_provider', start=init_data_provider_task)


@app.router('/start_learning', methods=['get'])
async def start_learning(request):
    query = request.url_data
    if not query['algo'] or query['algo'] not in algorithm_opts:
        return wsgi.WsgiResponse(400, json.dumps({'error': 'unknown algorithm type'}, indent=4))

    num_instances = 2
    if 'instances' in query:
        num_instances = int(query['instances'])

    activation = 'relu'
    if 'activation' in query:
        activation = query['activation']

    # run managers
    if query['algo'] in ['svm', 'all']:
        await pulsar.send('svm_manager_actor', 'run_svm_learning', {})
    if query['algo'] in ['bayes', 'all']:
        await pulsar.send('bayes_manager_actor', 'run_bayes_learning', {})
    if query['algo'] in ['network', 'all']:
        await pulsar.send('network_manager_actor', 'run_network_learning', {'instances': num_instances, 'activation': activation})

    info = await pulsar.send('arbiter', 'info')

    data = {
        'arbiter_info': info,
    }
    return wsgi.WsgiResponse(200, json.dumps(data, indent=4))


@app.router('/get_progress', methods=['get'])
async def get_progress(request):
    global arbiter, svm_manager_actor, bayes_manager_actor, network_manager_actor

    query = request.url_data
    if not query['algo'] or query['algo'] not in algorithm_opts:
        return wsgi.WsgiResponse(400, json.dumps({'error': 'unknown algorithm type'}, indent=4))

    progress = {}

    # run managers
    if query['algo'] in ['svm', 'all']:
        progress['svm'] = await pulsar.send('svm_manager_actor', 'get_svm_progress')
    if query['algo'] in ['bayes', 'all']:
        progress['bayes'] = await pulsar.send('bayes_manager_actor', 'get_bayes_progress')
    if query['algo'] in ['network', 'all']:
        progress['network'] = await pulsar.send('network_manager_actor', 'get_network_progress')

    data = progress
    return wsgi.WsgiResponse(200, json.dumps(data, indent=4))


@app.router('/stop_learning', methods=['get'])
async def stop_learning(request):
    query = request.url_data
    if not query['algo'] or query['algo'] not in algorithm_opts:
        return wsgi.WsgiResponse(400, json.dumps({'error': 'unknown algorithm type'}, indent=4))

    # kill manager actors
    if query['algo'] in ['svm', 'all']:
        await pulsar.send('svm_manager_actor', 'kill_svm_instances')
    if query['algo'] in ['bayes', 'all']:
        await pulsar.send('bayes_manager_actor', 'kill_bayes_instances')
    if query['algo'] in ['network', 'all']:
        await pulsar.send('network_manager_actor', 'kill_network_instances')

    info = await pulsar.send('arbiter', 'info')
    data = {
        'arbiter_info': info,
    }
    return wsgi.WsgiResponse(200, json.dumps(data, indent=4))


@app.router('/info', methods=['get'])
async def stop_manager(request):
    global arbiter

    info = await pulsar.get_actor().send(arbiter, 'info')
    data = {
        'arbiter_info': info,
    }
    return wsgi.WsgiResponse(200, json.dumps(data, indent=4))


@app.router('/data_sample', methods=['get'])
async def data_sample(request):
    query = request.url_data

    # default
    input_format = 'hot_vector'
    output_format = 'categorical'

    if 'input_format' in query:
        input_format = query['input_format']
    if 'output_format' in query:
        output_format = query['output_format']

    (X_data, Y_data) = await pulsar.send('data_provider', 'get_data_command',
                                         {'input_format': input_format, 'output_format': output_format})

    data = {
        'length': len(X_data),
        'X_data': str(X_data),
        'Y_data': str(Y_data),
    }
    return wsgi.WsgiResponse(200, json.dumps(data, indent=4))


@app.router('/data_info', methods=['get'])
async def data_info(request):
    global arbiter

    info = await pulsar.send('data_provider', 'data_provider_info')
    data = {
        'data_provider_info': info,
    }
    return wsgi.WsgiResponse(200, json.dumps(data, indent=4))


# start WSGI server
def wsgi_setup():
    return wsgi.WsgiHandler(middleware=[app], response_middleware=[AccessControl()])


def spawn_managers_wrapper(algo):
    asyncio.ensure_future(spawn_managers(algo))


def spawn_data_provider_wrapper():
    asyncio.ensure_future(spawn_data_provider())


if __name__ == '__main__':
    asyncio.get_event_loop().call_later(0.1, spawn_data_provider_wrapper)
    asyncio.get_event_loop().call_later(0.2, spawn_managers_wrapper, 'all')
    wsgi.WSGIServer(callable=wsgi_setup()).start()
