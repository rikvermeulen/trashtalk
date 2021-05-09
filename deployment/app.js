//filesystem
const fs = require('fs');

//tensorflow
const tf = require('@tensorflow/tfjs');
const tfnode = require('@tensorflow/tfjs-node');

//modules
const express = require('./server.js');
const labelClasses = require('./js/TopClasses');
const train = require('./js/node/train');

//model
const handler = tfnode.io.fileSystem("model/model.json");
const labels = require('./model/metadata.json').labels;

//raspi-gpio pins
const Gpio = require('onoff').Gpio

//button
let pushButton = new Gpio(20, 'in', "falling");

//led
const led_yellow = new Gpio(17, 'out');
const led_blue = new Gpio(27, 'out');
const led_red = new Gpio(23, 'out');
const led_green = new Gpio(22, 'out');
const led_pink = new Gpio(24, 'out');
const led_white = new Gpio(10, 'out');

//Camera
const { StillCamera } = require("pi-camera-connect");
const { SensorMode } = require("pi-camera-connect");
const { Codec } = require("pi-camera-connect");
const imagePath = "src/images/predict.png"

const stillCamera = new StillCamera({
  codec: Codec.PNG,
  sensormode: SensorMode.Mode7
});

//temp
// global.fetch = require("node-fetch");
// let Sound = require('node-aplay');


function load() {
  let start = express.server()
  console.log("Server Started")
}

function startTraining() {
  let training = train.train()
  console.log("Training Started")
}


pushButton.watch( async function (err, value) {
  if (err) { //if an error
    console.error('There was an error', err);
  return;
  }
  console.log("starting classify process")
    led_white.writeSync(1);
    const image = await stillCamera.takeImage()
    led_white.writeSync(0);
    fs.writeFileSync(imagePath, image);
    classify(image)
});

      

// async function takePhoto(){
//   stillCamera.takeImage().then(image => {
//       fs.writeFileSync(imagePath, image);
//       console.log("Picture taken")
//   });
// }

const classify = async (imageData) => {
  let decodedImage = tfnode.node.decodeImage(imageData, 3);
  console.log("Image decoded")
  decodedImage = tf.expandDims(decodedImage, axis=-0)
  decodedImage = tf.image.cropAndResize(image=decodedImage, boxes=[[.1,.1,.6,.6]], [0], crop_size=[224, 224])

  const model = await tfnode.loadLayersModel(handler);
  const logits = tfnode.tidy(() => {
    return model.predict(decodedImage);
  });

  const classes = await labelClasses.TopClasses(labels, logits, 3);
  const topResult = classes[0].className;
  console.log('My prediction is:', topResult);

  const led = await ledSelect(topResult);

  console.log('All predicitons are:', classes);

}

function ledSelect(topResult){
  switchLedOff();
    switch(topResult) {
        case 'plastic':
          led_yellow.writeSync(1);
          break;
        case 'cardboard':
          led_blue.writeSync(1);
          break;
        case 'glass':
          led_red.writeSync(1);
          break;
        case 'paper':
          led_green.writeSync(1);
          break;
        case 'metal':
          led_pink.writeSync(1);
          break;
        case 'human':
          led_white.writeSync(1);
          break;
        default: 
          console.log(`We don't know what it is`);
    }
}

function switchLedOff() {
  led_yellow.writeSync(0),
  led_blue.writeSync(0),
  led_red.writeSync(0),
  led_green.writeSync(0),
  led_pink.writeSync(0),
  led_white.writeSync(0); 
};

//testzone 

load()

function unexportOnClose() {
  switchLedOff()
  pushButton.unexport();
};

process.on('SIGINT', unexportOnClose);     //function ctrl+c