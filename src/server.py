import asyncio
import json
import pulsar.api as pulsar

import pulsar.apps.wsgi as wsgi
from pulsar.apps.wsgi import AccessControl

import classfier_managers.network_manager, classfier_managers.bayes_manager, classfier_managers.svm_manager

app = wsgi.Router('/')

arbiter = pulsar.arbiter()
algorithm_opts = ['svm', 'bayes', 'network', 'all']
svm_manager_actor, bayes_manager_actor, network_manager_actor = None, None, None


async def spawn_managers(algo='all'):
    global arbiter, svm_manager_actor, bayes_manager_actor, network_manager_actor

    if not svm_manager_actor and algo in ['svm', 'all']:
        svm_manager_actor = await pulsar.spawn(name='svm_manager_actor', aid='svm_manager_actor')

    if not bayes_manager_actor and algo in ['bayes', 'all']:
        bayes_manager_actor = await pulsar.spawn(name='bayes_manager_actor', aid='bayes_manager_actor')

    if not network_manager_actor and algo in ['network', 'all']:
        network_manager_actor = await pulsar.spawn(name='network_manager_actor', aid='network_manager_actor')


@app.router('/start_learning', methods=['get'])
async def start_learning(request):
    query = request.url_data
    if not query['algo'] or query['algo'] not in algorithm_opts:
        return wsgi.WsgiResponse(400, json.dumps({'error': 'unknown algorithm type'}, indent=4))

    # run managers
    if query['algo'] in ['svm', 'all']:
        await pulsar.send('svm_manager_actor', 'run_svm_learning', {})
    if query['algo'] in ['bayes', 'all']:
        await pulsar.send('bayes_manager_actor', 'run_bayes_learning', {})
    if query['algo'] in ['network', 'all']:
        await pulsar.send('network_manager_actor', 'run_network_learning', {})

    info = await pulsar.send('arbiter', 'info')

    data = {
        'arbiter_info': info,
    }
    return wsgi.WsgiResponse(200, json.dumps(data, indent=4))


@app.router('/get_progress', methods=['get'])
async def get_process(request):
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


# start WSGI server
def wsgi_setup():
    return wsgi.WsgiHandler(middleware=[app], response_middleware=[AccessControl()])


def spawn_managers_wrapper(algo):
    asyncio.ensure_future(spawn_managers(algo))


if __name__ == '__main__':
    asyncio.get_event_loop().call_later(1, spawn_managers_wrapper, 'all')
    wsgi.WSGIServer(callable=wsgi_setup()).start()
