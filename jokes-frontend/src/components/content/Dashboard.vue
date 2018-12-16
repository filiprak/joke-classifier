<template>
    <v-container fluid grid-list-md>
        <v-layout row wrap>

            <v-flex d-flex xs12 sm6 md4>
                <v-card class="flexcard">
                    <v-card-title primary class="title">Neural network</v-card-title>
                    <v-card-text class="grow">
                        <v-select
                            :items="network.representations"
                            :hide-details="true"
                            color="orange"
                            label="Joke representation"
                        ></v-select>
                        <v-select
                            :items="network.activationFunctions"
                            color="orange"
                            label="Activation function"
                        ></v-select>
                        <br>
                        <v-divider></v-divider>
                        <span class="font-weight-bold">
                            Running instances:
                            <v-btn flat icon color="orange" @click="updateProgress('network')">
                                <v-icon>cached</v-icon>
                            </v-btn>
                        </span>
                        <v-list two-line class="pa-0">
                            <template v-for="(item, index) in network.instances">
                                <v-list-tile :key="index">
                                    <v-list-tile-content>
                                        <span>{{item.id}}&nbsp;<v-icon small v-if="item.finished" color="green">done</v-icon></span>
                                        <v-list-tile-sub-title v-html="item.subtitle"></v-list-tile-sub-title>
                                        <v-progress-linear class="my-1" color="green" v-model="item.progress"></v-progress-linear>
                                    </v-list-tile-content>
                                </v-list-tile>

                            </template>
                            <span v-if="network.instances.length < 1" class="px-3">None</span>
                        </v-list>
                    </v-card-text>

                    <v-card-actions>
                        <v-btn flat color="green" v-on:click="startLearning('network')" :loading="network.running"><v-icon left>play_arrow</v-icon>Run</v-btn>
                        <v-btn flat color="red" v-on:click="stopLearning('network')" :loading="network.stopping"><v-icon left>stop</v-icon>Stop</v-btn>
                    </v-card-actions>
                </v-card>
            </v-flex>

            <v-flex d-flex xs12 sm6 md4>
                <v-card class="flexcard">
                    <v-card-title primary class="title">Naive Bayes</v-card-title>
                    <v-card-text class="grow">
                        <v-select
                                :items="bayes.representations"
                                :hide-details="true"
                                color="orange"
                                label="Joke representation"
                        ></v-select>
                        <br>
                        <v-divider></v-divider>
                        <span class="font-weight-bold">
                            Running instances:
                            <v-btn flat icon color="orange" @click="updateProgress('bayes')">
                                <v-icon>cached</v-icon>
                            </v-btn>
                        </span>
                        <v-list two-line class="pa-0">
                            <template v-for="(item, index) in bayes.instances">
                                <v-list-tile :key="index">
                                    <v-list-tile-content>
                                        <span>{{item.id}}&nbsp;<v-icon small v-if="item.finished" color="green">done</v-icon></span>
                                        <v-list-tile-sub-title v-html="item.subtitle"></v-list-tile-sub-title>
                                        <v-progress-linear class="my-1" color="green" v-model="item.progress"></v-progress-linear>
                                    </v-list-tile-content>
                                </v-list-tile>

                            </template>
                            <span v-if="bayes.instances.length < 1" class="px-3">None</span>
                        </v-list>
                    </v-card-text>

                    <v-card-actions>
                        <v-btn flat color="green" v-on:click="startLearning('bayes')" :loading="bayes.running"><v-icon left>play_arrow</v-icon>Run</v-btn>
                        <v-btn flat color="red" v-on:click="stopLearning('bayes')" :loading="bayes.stopping"><v-icon left>stop</v-icon>Stop</v-btn>
                    </v-card-actions>
                </v-card>
            </v-flex>

            <v-flex d-flex xs12 sm6 md4>
                <v-card class="flexcard">
                    <v-card-title primary class="title">SVM</v-card-title>
                    <v-card-text class="grow">
                        <v-select
                                :items="svm.representations"
                                :hide-details="true"
                                color="orange"
                                label="Joke representation"
                        ></v-select>
                        <br>
                        <v-divider></v-divider>
                        <span class="font-weight-bold">
                            Running instances:
                            <v-btn flat icon color="orange" @click="updateProgress('svm')">
                                <v-icon>cached</v-icon>
                            </v-btn>
                        </span>
                        <v-list two-line class="pa-0">
                            <template v-for="(item, index) in svm.instances">
                                <v-list-tile :key="index">
                                    <v-list-tile-content>
                                        <span>{{item.id}}&nbsp;<v-icon small v-if="item.finished" color="green">done</v-icon></span>
                                        <v-list-tile-sub-title v-html="item.subtitle"></v-list-tile-sub-title>
                                        <v-progress-linear class="my-1" color="green" v-model="item.progress"></v-progress-linear>
                                    </v-list-tile-content>
                                </v-list-tile>

                            </template>
                            <span v-if="svm.instances.length < 1" class="px-3">None</span>
                        </v-list>
                    </v-card-text>

                    <v-card-actions>
                        <v-btn flat color="green" v-on:click="startLearning('svm')" :loading="svm.running"><v-icon left>play_arrow</v-icon>Run</v-btn>
                        <v-btn flat color="red" v-on:click="stopLearning('svm')" :loading="svm.stopping"><v-icon left>stop</v-icon>Stop</v-btn>
                    </v-card-actions>
                </v-card>
            </v-flex>

            <div>&nbsp;</div>

            <v-flex d-flex xs12 sm12 md12>
                <v-card class="flexcard">
                    <v-card-title primary class="title">Master Classifier (work in progress)</v-card-title>
                    <v-card-text class="grow">
                        <br>
                        <v-divider></v-divider>
                        <br>
                        <span>Learning Progress</span>
                        <v-progress-linear class="my-1" color="green" v-model="master.progress"></v-progress-linear>
                    </v-card-text>

                    <v-card-actions>
                        <v-btn flat color="green" :loading="master.running"><v-icon left>play_arrow</v-icon>Run</v-btn>
                        <v-btn flat color="red" :loading="master.stopping"><v-icon left>stop</v-icon>Stop</v-btn>
                    </v-card-actions>
                </v-card>
            </v-flex>

        </v-layout>
        <ErrorModal v-bind:show="this.error.show"
                    v-bind:message="this.error.message"
                    v-bind:title="this.error.title"
                    @close="error.show = false"></ErrorModal>
    </v-container>
</template>

<script>
    import api from '../../api'
    import ErrorModal from '../ErrorModal'

    let pollProgress = null

    export default {
        name: "Dashboard",
        components: {
            ErrorModal
        },
        data() {
            return {
                serverInfo: null,
                lorem: 'Lorem',

                network: {
                    running: false,
                    stopping: false,
                    representations: ['bag-of-words', 'n-grams'],
                    activationFunctions: ['relu', 'softmax', 'tanh'],
                    instances: []
                },
                bayes: {
                    running: false,
                    stopping: false,
                    representations: ['bag-of-words', 'n-grams'],
                    instances: []
                },
                svm: {
                    running: false,
                    stopping: false,
                    representations: ['bag-of-words', 'n-grams'],
                    instances: []
                },

                master: {
                    running: false,
                    stopping: false,
                    progress: 33
                },

                error: {
                    title: '',
                    message: '',
                    show: false,
                }
            }
        },
        methods: {
            startLearning(algo) {
                api.get('/start_learning', { params: { algo: algo } }).then((res) => {
                    switch (algo) {
                        case 'network':
                            this.network.running = true
                            this.network.stopping = false
                            break
                        case 'svm':
                            this.svm.running = true
                            this.svm.stopping = false
                            break
                        case 'bayes':
                            this.bayes.running = true
                            this.bayes.stopping = false
                            break
                    }

                }, this.err)
            },
            stopLearning(algo) {
                switch (algo) {
                    case 'network': this.network.stopping = true; break
                    case 'svm': this.svm.stopping = true; break
                    case 'bayes': this.bayes.stopping = true; break
                }
                api.get('/stop_learning', { params: { algo: algo } }).then((res) => {
                    switch (algo) {
                        case 'network': this.network.running = false; break
                        case 'svm': this.svm.running = false; break
                        case 'bayes': this.bayes.running = false; break
                    }
                }, this.err).then(() => {
                    switch (algo) {
                        case 'network': this.network.stopping = false; break
                        case 'svm': this.svm.stopping = false; break
                        case 'bayes': this.bayes.stopping = false; break
                    }
                })

            },

            startProgressPolling() {
                if (!pollProgress) {
                    pollProgress = setInterval(() => {
                        this.updateProgress()
                    }, 1000)
                }
            },
            stopProgressPolling() {
                pollProgress ? clearInterval(pollProgress) : null
                pollProgress = null
            },

            updateProgress(algo = 'all') {
                api.get('/get_progress', { params: { algo: algo } }).then((res) => {
                    let net = res.data.network
                    let svm = res.data.svm
                    let bayes = res.data.bayes
                    if (net) {
                        this.network.instances = Object.keys(net).map((key, idx) => {
                            return { id: key, subtitle: '', progress: net[key], finished: net[key] > 100 }
                        })
                        let finished = true;
                        for (let key in net) { if (net[key] < 100) { finished = false } }
                        this.network.running = finished ? false : this.network.running
                    }
                    if (bayes) {
                        this.bayes.instances = Object.keys(bayes).map((key, idx) => {
                            return { id: key, subtitle: '', progress: bayes[key], finished: bayes[key] > 100 }
                        })
                        let finished = true;
                        for (let key in bayes) { if (bayes[key] < 100) { finished = false } }
                        this.bayes.running = finished ? false : this.bayes.running
                    }
                    if (svm) {
                        this.svm.instances = Object.keys(svm).map((key, idx) => {
                            return { id: key, subtitle: '', progress: svm[key], finished: svm[key] > 100 }
                        })
                        let finished = true;
                        for (let key in svm) { if (svm[key] < 100) { finished = false } }
                        this.svm.running = finished ? false : this.svm.running
                    }

                }, this.err)
            },

            err(err, title) {
                this.error.title = title
                this.error.message = JSON.stringify(err, null, 2)
                this.error.show = true
            }
        },
        mounted() {
            api.get('/info').then((res) => {
                this.serverInfo = JSON.stringify(res)
            }).then(() => {
                this.startProgressPolling()
            })
        },
        beforeDestroy() {
            this.stopProgressPolling()
        }
    }
</script>

<style scoped>
.flexcard {
    display: flex;
    flex-direction: column;
}
</style>