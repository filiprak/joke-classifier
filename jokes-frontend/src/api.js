import axios from "axios";

export default axios.create({
    baseURL: 'http://localhost:8060/',
    timeout: 30000,
})
