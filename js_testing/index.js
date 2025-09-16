import { callPythonScript, form, inputField, resultFactorial } from './factorialSubmit.js';
import { dropArea, resultTesseract, callTesseractScript} from './drop.js';
const url = 'http://localhost:3000';

dropArea.addEventListener('drop',async (e) => {
    e.preventDefault();

    const file = e.dataTransfer.files[0];
    console.log("The file recieved is : ", file.name)

    if(!file){
        alert("No file dropped")
        return;
    }

    const res = await callTesseractScript(url, file);
    console.log(res.result)
})

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const number = parseInt(inputField.value);
    console.log("The number received is ", number);

    if (isNaN(number) || number <= 0) {
        alert("Enter a valid number (must be greater than 0)");
        inputField.value = '';
        return;
    }

    inputField.value = '';

    const res = await callPythonScript(url, number);
    console.log(res.factorial);

    resultFactorial.innerHTML = `<h4>The factorial of ${number} is ${res.factorial}</h4>`;
});
