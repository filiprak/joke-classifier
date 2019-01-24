import Vue from "vue";
import VueRouter from 'vue-router';

import Dashboard from './components/content/Dashboard'
import Settings from './components/content/Settings'
import Jokes from './components/content/Jokes'

Vue.use(VueRouter);

let router = new VueRouter({
    routes: [
        {
            path: '/dashboard',
            name: 'Dashboard',
            component: Dashboard,
        },
        {
            path: '/settings',
            name: 'Settings',
            component: Settings,
        },
        {
            path: '/jokes',
            name: 'Jokes',
            component: Jokes,
        },
        { path: '*', redirect: '/dashboard' }
    ]
});

export default router;