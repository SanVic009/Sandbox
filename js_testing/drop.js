export const dropArea = document.querySelector("#dropArea")

export async function callTesseractScript(url, address){
    try{
        if (address){
            throw new Error("Invalid Input")
        }

        const response = await fetch(`${url}/${address}`);
        if(!response.ok){
            throw new Error(`Response status: ${response.status}`)
        }

        const json = await response.json()
        return json;
    } catch(error){
        return {error: error.message}
    }
}