<template>
    <div class="text-xs-center">
        <v-dialog
                v-model="$props.show"
                width="600"
        >
            <v-card>
                <v-card-title
                        class="headline red"
                        primary-title
                >
                    <v-icon x-large>error_outline</v-icon>
                    &nbsp;{{this.computedTitle}}
                </v-card-title>

                <v-card-text>
                    {{this.computedMessage}}
                </v-card-text>

                <v-divider></v-divider>

                <v-card-actions>
                    <v-spacer></v-spacer>
                    <v-btn v-for="btn in this.computedButtons"
                            :color="btn.color"
                            :key="btn.label"
                            flat
                            @click="btn.onclick"
                    >
                        {{ btn.label }}
                    </v-btn>
                </v-card-actions>
            </v-card>
        </v-dialog>
    </div>
</template>

<script>
    export default {
        name: "ErrorModal",
        props: {
            show: Boolean,
            title: {
                type: String,
                default: 'Something went wrong...'
            },
            message: {
                type: String,
                default: 'Unknown error occurred'
            },
            buttons: {
                type: Array,
                default: () => [{
                    color: 'orange',
                    label: 'OK',
                    onclick: 'close'
                }]
            }
        },
        computed: {
            computedButtons() {
                return this.$props.buttons.map((btn) => {
                    btn.onclick = btn.onclick == 'close' ? this.close : btn.onclick
                    return btn
                })
            },
            computedMessage() {
                return this.$props.message || 'Unknown error occurred'
            },
            computedTitle() {
                return this.$props.title || 'Something went wrong...'
            }
        },
        methods: {
            close() {
                this.$emit('close')
            }
        },
    }
</script>

<style scoped>

</style>