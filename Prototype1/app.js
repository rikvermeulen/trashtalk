const tfnode = require('@tensorflow/tfjs-node');
const fs = require('fs');

const { StillCamera } = require("pi-camera-connect");
const { SensorMode } = require("pi-camera-connect");
const { Codec } = require("pi-camera-connect");

const tf = require('@tensorflow/tfjs');

const Gpio = require('onoff').Gpio

const handler = tfnode.io.fileSystem("model/model.json");
const metadata = require('./model/metadata.json');
const labels = metadata.labels;

let pushButton = new Gpio(21, 'in', "falling");

const led_yellow = new Gpio(17, 'out');
const led_blue = new Gpio(27, 'out');
const led_red = new Gpio(23, 'out');
const led_green = new Gpio(22, 'out');

const stillCamera = new StillCamera({
    codec: Codec.PNG,
    sensormode: SensorMode.Mode7
    // rotation: Rotation.Rotate0,
});

pushButton.watch(function (err, value) {
  if (err) {
    console.error('There was an error', err);
  return;
  }
  console.log("starting classify process")
  takePhoto()
});

function takePhoto(){
  const imagePath = "images/predict.png"
  stillCamera.takeImage().then(image => {
      fs.writeFileSync(imagePath, image);
      console.log("Picture taken")
      classify(imagePath)
  });
}

const classify = async (imagePath) => {
  let image = fs.readFileSync(imagePath);
  let decodedImage = tfnode.node.decodeImage(image, 3);
  decodedImage = tf.expandDims(decodedImage, axis=-0)
  decodedImage = tf.image.cropAndResize(image=decodedImage, boxes=[[.1,.1,.6,.6]], [0], crop_size=[224, 224])

  const model = await tfnode.loadLayersModel(handler);
  const logits = tfnode.tidy(() => {
    return model.predict(decodedImage);
  });

  const classes = await getTopKClasses(labels, logits, 3);
  const topResult = classes[0].className;
  console.log('My prediction is:', topResult);

  const led = await ledSelect(topResult);

  console.log('All predicitons are:', classes);

}

async function getTopKClasses(labels, logits, topK = 3) {
  const values = await logits.data();

  return tfnode.tidy(() => {
    topK = Math.min(topK, values.length);

    const valuesAndIndices = [];
    for (let i = 0; i < values.length; i++) {
      valuesAndIndices.push({ value: values[i], index: i });
    }
    valuesAndIndices.sort((a, b) => {
      return b.value - a.value;
    });
    const topkValues = new Float32Array(topK);
    const topkIndices = new Int32Array(topK);
    for (let i = 0; i < topK; i++) {
      topkValues[i] = valuesAndIndices[i].value;
      topkIndices[i] = valuesAndIndices[i].index;
    }

    const topClassesAndProbs = [];
    for (let i = 0; i < topkIndices.length; i++) {
      topClassesAndProbs.push({
        className: labels[topkIndices[i]],
        probability: topkValues[i],
      });
    }
    return topClassesAndProbs;
  });
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
        case 'plastic':
          led_red.writeSync(1);
          break;
        case 'paper':
          led_green.writeSync(1);
          break;
        default: 
          console.log(`We don\'t have ${topResult} color LED`);
    }
}

function switchLedOff() {
  led_yellow.writeSync(0);
  led_blue.writeSync(0);
  led_red.writeSync(0);
  led_green.writeSync(0); 
};

function unexportOnClose() { 
  switchLedOff()
  pushButton.unexport(); 
};

process.on('SIGINT', unexportOnClose); 

//testzone 
async function train(){
  labelArray = [0, 1]
  const imagePath = "images/predict.png"
  const imageBuffer = await fs.readFileSync(imagePath);
  tensorFeature1 = tfnode.node.decodeImage(imageBuffer)
  tensorFeature2 = tfnode.node.decodeImage(imageBuffer)
  
  // create an array of all the features
  tensorFeatures = tf.stack([tensorFeature1, tensorFeature2])
  tensorLabels = tf.oneHot(tf.tensor1d(labelArray, 'int32'), 3);
  tensorFeatures = tf.image.cropAndResize(image=tensorFeatures, boxes=[[.1,.1,.6,.6],[.1,.1,.6,.6]], [0, 1], crop_size=[224, 224])
  console.log(tensorFeatures)

  const trainmodel = tf.sequential();
  trainmodel.add(tf.layers.conv2d({
    inputShape: [224, 224, 3],
    filters: 32,
    kernelSize: 3,
    activation: 'relu',
  }));
  trainmodel.add(tf.layers.flatten()),
  trainmodel.add(tf.layers.dense({units: 3, activation: 'softmax'}));
  trainmodel.compile({optimizer: 'sgd', loss: 'meanSquaredError'})
  trainmodel.fit(tensorFeatures, tensorLabels)
}

// train()
