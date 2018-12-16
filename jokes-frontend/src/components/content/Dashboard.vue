<template>
    <v-container fluid grid-list-md>
        <v-layout row wrap>
            <v-flex d-flex xs12 sm6 md4>
                <v-card dark>
                    <v-card-title primary class="title">Neural network</v-card-title>
                    <v-card-text>
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
                        <span>Running instances:</span>
                        <v-list two-line>
                            <template v-for="(item, index) in network.instances">
                                <v-list-tile :key="index">
                                    <v-list-tile-content>
                                        <v-list-tile-sub-title v-html="item.title"></v-list-tile-sub-title>
                                        <v-progress-linear color="green" v-model="item.progress"></v-progress-linear>
                                    </v-list-tile-content>
                                </v-list-tile>
                            </template>
                        </v-list>
                    </v-card-text>

                    <v-card-actions>
                        <v-btn flat color="green"><v-icon left>play_arrow</v-icon>Run</v-btn>
                        <v-btn flat color="red"><v-icon left>stop</v-icon>Stop</v-btn>
                    </v-card-actions>
                </v-card>
            </v-flex>
            <v-flex d-flex xs12 sm6 md4>
                <v-card dark>
                    <v-card-title primary class="title">Naive Bayes</v-card-title>
                    <v-card-text>{{ lorem }}</v-card-text>
                </v-card>
            </v-flex>
            <v-flex d-flex xs12 sm6 md4>
                <v-card dark>
                    <v-card-title primary class="title">SVM</v-card-title>
                    <v-card-text>{{ lorem }}</v-card-text>
                </v-card>
            </v-flex>
        </v-layout>
    </v-container>
</template>

<script>
    import api from '../../api'

    export default {
        name: "Dashboard",
        data() {
            return {
                lorem: 'Lorem',
                network: {
                    representations: ['bag-of-words', 'n-grams'],
                    activationFunctions: ['relu', 'softmax', 'tanh'],
                    instances: [
                        { title: 'network_instance0', subtitle: '', progress: 35},
                        { title: 'network_instance1', subtitle: '', progress: 77},
                    ]
                },
                bayes: {

                },
                svm: {

                },

                valueDeterminate: 35
            }
        },
        methods: {

        },
        mounted: () => {
            api.get('/info').then((res) => {
                this.lorem = JSON.stringify(res)
            })
        }
    }
</script>

<style scoped>

</style>