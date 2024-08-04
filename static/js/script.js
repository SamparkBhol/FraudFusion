document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('file-input');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    document.getElementById('result').innerHTML = 'Prediction: ' + JSON.stringify(result);
});
