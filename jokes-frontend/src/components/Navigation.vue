<template>
    <v-navigation-drawer app fixed v-model="$props.menuOpened" clipped>
        <v-list dense class="pt-0">
            <v-list-tile
                    v-for="item in items"
                    :key="item.path"
                    :to="{path: item.path}"
                    active-class="highlighted"
            >
                <v-list-tile-action class="pl-2">
                    <v-icon>{{ item.icon }}</v-icon>
                </v-list-tile-action>

                <v-list-tile-content>
                    <v-list-tile-title>{{ item.title }}</v-list-tile-title>
                </v-list-tile-content>

            </v-list-tile>
        </v-list>
    </v-navigation-drawer>
</template>

<script>
    export default {
        name: "Navigation",
        props: {
            menuOpened: Boolean
        },
        data() {
            return {
                items: [
                    {title: 'Dashboard', icon: 'dashboard', path: 'dashboard'},
                    {title: 'Example Jokes', icon: 'tag_faces', path: 'jokes'},
                    //{title: 'Settings', icon: 'settings', path: 'settings'},
                ]
            }
        },
    }
</script>

<style>
    .highlighted {
        color: orange;
    }
</style>