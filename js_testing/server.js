import express from 'express';
import cors from 'cors';
import { spawn } from 'child_process';
 
const app = express();
const port = 3000;

app.use(cors());

app.use(cors({
    origin: ['http://127.0.0.1:5500', 'http://localhost:5500'], 
    methods: ['GET', 'POST'], 
    allowedHeaders: ['Content-Type']
}));

app.get('/tesseract/:string', (req, res) =>{
    const address = req.params.string;
    console.log("Address recieved :", address)
    runTesseractScript('tesseract.py', [address], (err, result)=>{
        if (err){
            console.error(`Error: ${err}`)
            res.status(500).json({error: err});
        }
        try{
            const response = JSON.parse(result);
            res.json(response)
        }
        catch(parseErro){
            console.error(`JSON parse error: ${parseErro}`)
            res.status(500).json({error:"Invalid response"})
        }
    })
})

app.get('/factorial/:number', (req, res) =>{
    const number = req.params.number;
    console.log("Number recieved :", number)
    runFactorialScript('script.py', [number], (err, result) => {
        if (err){
            console.error(`Error: ${err}`)
            res.status(500).json({error: err});
        }
        try{
            const response = JSON.parse(result);
            res.json(response)
        }
        catch(parseErro){
            console.error(`JSON parse error: ${parseErro}`)
            res.status(500).json({error:"Invalid response"})
        }
    })
})

app.listen(port, () =>{
    console.log(`Server running at http://localhost:${port}`)
})

function runFactorialScript(scriptPath, args, callback){
    const pythonProcess = spawn('python', [scriptPath].concat(args));

    let data = ''
    pythonProcess.stdout.on('data', (chunk) =>{
        data += chunk.toString();
    });

    pythonProcess.stderr.on('data', (error) =>{
        console.error(`stderr: ${error}`);
    });

    pythonProcess.on("close", (code)=>{
        if(code != 0){
            console.log(`Python script exited with code ${code}`);
            callback(`Error: script exited with code ${code}`, null);
        } else{
            console.log('Python script executed sucessflly');
            callback(null, data.trimEnd());
        }
    });
}

function runTesseractScript(scriptPath, args, callback){
    const pythonProcess = spawn('python', [scriptPath].concat(args));

    let data = ''
    pythonProcess.stdout.on('data', (chunk) =>{
        data += chunk.toString();
    });

    pythonProcess.stderr.on('data', (error)=>{
        console.error(`stderr: ${error}`);
    });

    pythonProcess.on('close', (code) =>{
        if(code !=0){
            console.log(`Python script exited with code ${code}`);
            callback(`Error: script exited with code ${code}`, null);
        } else{
            console.log('Python script executed sucessflly');
            callback(null, data.trimEnd());
        }
    
    })
}