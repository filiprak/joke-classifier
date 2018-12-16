import Vue from "vue";
import VueRouter from 'vue-router';

import Dashboard from './components/content/Dashboard'
import Settings from './components/content/Settings'

Vue.use(VueRouter);

let router = new VueRouter({
    mode: 'history',
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
        { path: '*', redirect: '/dashboard' }
    ]
});

export default router;