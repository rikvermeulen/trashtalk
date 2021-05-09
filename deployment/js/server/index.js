const handler = "./model/model.json";
let metadata = 'http://localhost:8080/model/metadata.json'
let labels;

const synth = window.speechSynthesis;

//speech
function speak(text){
    if (synth.speaking){
        console.log('ssssttt')
        return
    }

    if (text !== '') {
        let utterThis = new SpeechSynthesisUtterance(text)
        synth.speak(utterThis)
    }
}

//predict
const fileButton = document.querySelector("#file")
const images = document.getElementById("selected-image")

fileButton.addEventListener("change", (event) => loadFile(event))

function loadFile(event) {
    images.src = URL.createObjectURL(event.target.files[0])
    console.log("image Loaded in DOM")
}

fetch(metadata).then(
    function (u) { return u.json(); }
).then(
    function (json) {
        labels = json.labels;
    }
)

function startPredict() {
    try {
        let pre_image = tf.browser.fromPixels(images, 3)
        classify(pre_image)
    }
    catch (err) {
        document.getElementById("error").innerHTML = "Upload an image first!"
    }
}


//train
const filesButton = document.querySelector("#files")
const imagess = document.getElementById("trainlist").getElementsByTagName("img")
const imagesss = document.getElementById("selected-images")

const button = document.getElementById('traintBtn')

filesButton.addEventListener("change", (event) => loadFiles(event))
// imagesss.addEventListener("load", ()=>trainModel())

function loadFiles(event) {
    console.log(imagess)
    for (var i = 0; i < imagess.length; i++) {
        imagess[i].src = URL.createObjectURL(event.target.files[i])
    }
}

function checkFiles(files) {
    if (files.length < 10) {
        button.disabled = true
        document.getElementById("errorTrain").innerHTML = "You need 10 or more images to start training!"

    } else {
        button.disabled = false
        document.getElementById("errorTrain").innerHTML = ""
    }
}