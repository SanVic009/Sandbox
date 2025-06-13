export const form = document.querySelector('.form');
export const button = document.querySelector('#submitButton');
export const inputField = document.querySelector('#inputField');
export const result = document.querySelector('#result');

console.log(form)

export async function callPythonScript(url, number) {
    try {
        if (isNaN(number) || number <= 0) {
            throw new Error("Invalid input");
        }

        const response = await fetch(`${url}/factorial/${number}`);
        if (!response.ok) {
            throw new Error(`Response status: ${response.status}`);
        }

        const json = await response.json();
        return json;
    } catch (error) {
        console.log(error.message);
        return { error: error.message };
    }
}
