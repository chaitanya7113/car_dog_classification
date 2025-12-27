const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const result = document.getElementById("result");

// Show image preview
imageInput.addEventListener("change", function () {
    const file = imageInput.files[0];
    const reader = new FileReader();

    reader.onload = function () {
        preview.src = reader.result;
        preview.style.display = "block";
    };
    reader.readAsDataURL(file);
});

function predict() {
    const file = imageInput.files[0];

    if (!file) {
        alert("Please select an image first");
        return;
    }

    const formData = new FormData();
    formData.append("image", file);

    // BACKEND API CALL
    fetch("https://car-dog-classification-1.onrender.com/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        result.innerHTML = `
            Prediction: <b>${data.label}</b><br>
            Confidence: ${(data.confidence * 100).toFixed(2)}%
        `;
    })
    .catch(error => {
        console.error(error);
        result.innerHTML = "Error connecting to server";
    });
}
