import {createClient} from '@supabase/supabase-js'
import dotenv from 'dotenv';

dotenv.config();

const supabase_url = process.env.SUPABASE_PROJECT_URL
const service_key = process.env.SERVICE_KEY

const supabase = createClient(supabase_url, service_key)

async function insertPlanets() {
    const {data, error} = await supabase
    .from('planets')
    .insert([{ name:'MARS'}]);

    if (error){
        console.error('Insert error: ', error)
    } else{
        console.log('Insert successful: ', data)
    }
}

insertPlanets();