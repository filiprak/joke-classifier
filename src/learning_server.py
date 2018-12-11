import json
import pulsar.api as pulsar

import pulsar.apps.wsgi as wsgi

from classfiers.bayes import run_bayes_instance
from classfiers.network import run_network_instance
from classfiers.svm import run_svm_instance
from utils import is_int

app = wsgi.Router('/')

arbiter = pulsar.arbiter()
algorithm_opts = ['svm', 'bayes', 'network']
svm_instances, bayes_instances, network_instances = [], [], []


@app.router('/run_learning', methods=['get'])
async def run_learning(request):
    global arbiter, svm_instances, bayes_instances, network_instances

    query = request.url_data
    if not query['algo'] or query['algo'] not in algorithm_opts:
        return wsgi.WsgiResponse(400, json.dumps({'error': 'unknown algorithm type'}, indent=4))

    instances = int(query['instances'])

    # kill previously running instances
    if query['algo'] == 'svm':
        for instance_proxy in svm_instances:
            instance_proxy.kill()
        svm_instances = []
    elif query['algo'] == 'bayes':
        for instance_proxy in bayes_instances:
            instance_proxy.kill()
        bayes_instances = []
    elif query['algo'] == 'network':
        for instance_proxy in network_instances:
            instance_proxy.kill()
        network_instances = []

    if len(svm_instances) < 5 and query['algo'] == 'svm':
        for i in range(min(instances, 5 - len(svm_instances))):
            instance_proxy = await pulsar.spawn(name='svm_instance')
            pulsar.send(instance_proxy, 'run', run_svm_instance, args={})
            svm_instances.append(instance_proxy)

    elif len(bayes_instances) < 5 and query['algo'] == 'bayes':
        for i in range(min(instances, 5 - len(bayes_instances))):
            instance_proxy = await pulsar.spawn(name='bayes_instance')
            pulsar.send(instance_proxy, 'run', run_bayes_instance, args={})
            bayes_instances.append(instance_proxy)

    elif len(network_instances) < 5 and query['algo'] == 'network':
        for i in range(min(instances, 5 - len(network_instances))):
            instance_proxy = await pulsar.spawn(name='network_instance')
            pulsar.send(instance_proxy, 'run', run_network_instance, args={})
            network_instances.append(instance_proxy)

    info = await pulsar.get_actor().send(arbiter, 'info')

    data = {
        'arbiter_info': info,
    }
    return wsgi.WsgiResponse(200, json.dumps(data, indent=4))


@app.router('/stop_learning', methods=['get'])
async def stop_learning(request):
    global arbiter, svm_instances, bayes_instances, network_instances

    query = request.url_data
    if not query['algo'] or query['algo'] not in algorithm_opts:
        return wsgi.WsgiResponse(400, json.dumps({'error': 'unknown algorithm type'}, indent=4))

    # kill all algorithm instances
    if query['algo'] == 'svm':
        for instance_proxy in svm_instances:
            instance_proxy.kill()
        svm_instances = []
    elif query['algo'] == 'bayes':
        for instance_proxy in bayes_instances:
            instance_proxy.kill()
        bayes_instances = []
    elif query['algo'] == 'network':
        for instance_proxy in network_instances:
            instance_proxy.kill()
        network_instances = []

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
