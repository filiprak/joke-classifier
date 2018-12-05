import Vue from "vue";
import VueRouter from 'vue-router';

import Dashboard from './components/content/Dashboard'
import Settings from './components/content/Settings'
import HelloWorld from './components/HelloWorld'

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
        {
            path: '/helloworld',
            name: 'HelloWorld',
            component: HelloWorld,
        },
        { path: '*', redirect: '/helloworld' }
    ]
});

export default router;