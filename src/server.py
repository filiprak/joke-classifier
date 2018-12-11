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


@app.router('/start_manager', methods=['get'])
async def start_manager(request):
    global arbiter, svm_manager_actor, bayes_manager_actor, network_manager_actor

    query = request.url_data
    if not query['algo'] or query['algo'] not in algorithm_opts:
        return wsgi.WsgiResponse(400, json.dumps({'error': 'unknown algorithm type'}, indent=4))

    if not svm_manager_actor and query['algo'] in ['svm', 'all']:
        svm_manager_actor = await pulsar.spawn(name='svm_manager_actor', aid='svm_manager_actor')

    if not bayes_manager_actor and query['algo'] in ['bayes', 'all']:
        bayes_manager_actor = await pulsar.spawn(name='bayes_manager_actor', aid='bayes_manager_actor')

    if not network_manager_actor and query['algo'] in ['network', 'all']:
        network_manager_actor = await pulsar.spawn(name='network_manager_actor', aid='network_manager_actor')

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


# start WSGI server
def wsgi_setup():
    return wsgi.WsgiHandler([app])


if __name__ == '__main__':
    wsgi.WSGIServer(callable=wsgi_setup()).start()
