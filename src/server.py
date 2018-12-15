import asyncio
import json
import pulsar.api as pulsar

import pulsar.apps.wsgi as wsgi

from classfier_managers.bayes_manager import run_bayes_manager
from classfier_managers.network_manager import run_network_manager
from classfier_managers.svm_manager import run_svm_manager

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


@app.router('/start_manager', methods=['get'])
async def start_manager(request):
    global arbiter, svm_manager_actor, bayes_manager_actor, network_manager_actor

    query = request.url_data
    if not query['algo'] or query['algo'] not in algorithm_opts:
        return wsgi.WsgiResponse(400, json.dumps({'error': 'unknown algorithm type'}, indent=4))

    # run managers
    if query['algo'] in ['svm', 'all']:
        pulsar.send(svm_manager_actor, 'run', run_svm_manager, args={})
    if query['algo'] in ['bayes', 'all']:
        pulsar.send(bayes_manager_actor, 'run', run_bayes_manager, args={})
    if query['algo'] in ['network', 'all']:
        pulsar.send(network_manager_actor, 'run', run_network_manager, args={})

    info = await pulsar.get_actor().send(arbiter, 'info')

    data = {
        'arbiter_info': info,
    }
    return wsgi.WsgiResponse(200, json.dumps(data, indent=4))


@app.router('/get_progress', methods=['get'])
async def start_manager(request):
    global arbiter, svm_manager_actor, bayes_manager_actor, network_manager_actor

    query = request.url_data
    if not query['algo'] or query['algo'] not in algorithm_opts:
        return wsgi.WsgiResponse(400, json.dumps({'error': 'unknown algorithm type'}, indent=4))

    progress = {}

    # run managers
    if query['algo'] in ['svm', 'all']:
        progress['svm'] = await pulsar.send(svm_manager_actor, 'get_progress')
    if query['algo'] in ['bayes', 'all']:
        progress['svm'] = await pulsar.send(bayes_manager_actor, 'get_progress')
    if query['algo'] in ['network', 'all']:
        progress['svm'] = await pulsar.send(network_manager_actor, 'get_progress')

    data = {
        'progress': progress,
    }
    return wsgi.WsgiResponse(200, json.dumps(data, indent=4))


@app.router('/stop_manager', methods=['get'])
async def stop_manager(request):
    global arbiter, svm_manager_actor, bayes_manager_actor, network_manager_actor

    query = request.url_data
    if not query['algo'] or query['algo'] not in algorithm_opts:
        return wsgi.WsgiResponse(400, json.dumps({'error': 'unknown algorithm type'}, indent=4))

    # kill manager actors
    if svm_manager_actor and query['algo'] in ['svm', 'all']:
        svm_manager_actor.kill()
        svm_manager_actor = None
    if bayes_manager_actor and query['algo'] in ['bayes', 'all']:
        bayes_manager_actor.kill()
        bayes_manager_actor = None
    if network_manager_actor and query['algo'] in ['network', 'all']:
        network_manager_actor.kill()
        network_manager_actor = None

    info = await pulsar.get_actor().send(arbiter, 'info')
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
    return wsgi.WsgiHandler([app])


def spawn_manages_wrapper(algo):
    asyncio.ensure_future(spawn_managers(algo))


if __name__ == '__main__':
    asyncio.get_event_loop().call_later(0.1, spawn_manages_wrapper, 'all')
    wsgi.WSGIServer(callable=wsgi_setup()).start()
