<template>
    <v-app dark>

        <v-toolbar app flat clipped-left>
            <v-toolbar-side-icon v-on:click="toggleMenu"></v-toolbar-side-icon>
            <v-toolbar-title class="headline text-uppercase">
                <span>Vuetify</span>
                <span class="font-weight-light">MATERIAL DESIGN</span>
            </v-toolbar-title>

            <v-spacer></v-spacer>

            <v-btn icon>
                <v-icon>search</v-icon>
            </v-btn>

            <v-btn icon>
                <v-icon>favorite</v-icon>
            </v-btn>

            <v-btn icon>
                <v-icon>more_vert</v-icon>
            </v-btn>

        </v-toolbar>

        <v-content>
            <Navigation v-bind:menu-opened="menuOpened"></Navigation>
            <v-container fluid>
                <router-view></router-view>
            </v-container>
        </v-content>
    </v-app>
</template>

<script>
    import Navigation from './components/Navigation'

    export default {
        name: 'App',
        components: {
            Navigation
        },
        data() {
            return {
                menuOpened: true,
            }
        },
        methods: {
            toggleMenu() {
                this.menuOpened = !this.menuOpened;
            }
        }
    }
</script>
