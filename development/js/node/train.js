//filesystem
const fs = require('fs');

//tensorflow
const tf = require('@tensorflow/tfjs');
const tfnode = require('@tensorflow/tfjs-node');

const { StillCamera } = require("pi-camera-connect");
const { SensorMode } = require("pi-camera-connect");
const { Codec } = require("pi-camera-connect");

const labels = ["plastic","cardboard","glass","paper","metal","human"]

const stillCamera = new StillCamera({
  codec: Codec.PNG,
  sensormode: SensorMode.Mode7
});


const numOfClasses = 6
const epochsvalue = 3
const batchSize = 10;
const imageWidth = 28;
const imageHeight = 28;
const imageChannels = 3;

module.exports = {
    train: function train() {
    
    const genFeatureTensor = async imagePath => { 
      let image = await stillCamera.takeImage()
        fs.writeFileSync(imagePath, image);
        image = tfnode.node.decodeImage(image)
        image = tf.expandDims(image, axis=-0)
        image = await tf.image.resizeBilinear(image, [imageWidth, imageHeight] )
        return image
      }  
      
    
      const labelArray = indice => Array.from({length: 1}, (_, k) => k === indice ? 1 : 0)
      
    async function* dataGenerator() {
        const numElements = 4;
        let index = 0;
        while (index < numElements) {
          const imagePath = "./images/train/training"+[index]+".png"
          const feature = await genFeatureTensor(imagePath) ;
          const label = tf.tensor1d(labelArray(labels))
          index++;
          yield {xs: feature, ys: label};
        }
      
      }
      
    
      const trainsModel = async function (model, trainingData, epochs=epochsvalue) {
        const options = {
          epochs: epochs,
          verbose: 0,
          callbacks: {
            onEpochBegin: async (epoch, logs) => {
              console.log(`Epoch ${epoch + 1} of ${epochs} ...`)
            },
            onEpochEnd: async (epoch, logs) => {
              console.log(`  train-set loss: ${logs.loss.toFixed(4)}`)
              console.log(`  train-set accuracy: ${logs.acc.toFixed(4)}`)
            }
          }
        };
      
        return await model.fitDataset(trainingData, options);
      };
    
    
      const buildModel = function () {
        const model = tf.sequential();
        model.add(tf.layers.conv2d({
          inputShape: [imageWidth, imageHeight, imageChannels],
          filters: 8,
          kernelSize: 5,
          padding: 'same',
          activation: 'relu'
        }));
        model.add(tf.layers.maxPooling2d({
          poolSize: 2,
          strides: 2
        }));
        model.add(tf.layers.conv2d({
          filters: 16,
          kernelSize: 5,
          padding: 'same',
          activation: 'relu'
        }));
        model.add(tf.layers.maxPooling2d({
          poolSize: 3,
          strides: 3
        }));
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({
          units: numOfClasses,
          activation: 'softmax'
        }));
      
        model.compile({
          optimizer: tf.train.adam(),
          loss: 'categoricalCrossentropy',
          metrics: ['accuracy']
        });
      
        return model;
      };
    
      const evaluateModel = async function (model, testingData) {
        const result = await model.evaluateDataset(testingData);
        const testLoss = result[0].dataSync()[0];
        const testAcc = result[1].dataSync()[0];
      
        console.log(`  test-set loss: ${testLoss.toFixed(4)}`);
        console.log(`  test-set accuracy: ${testAcc.toFixed(4)}`);
      };

      const run = async function () {
        // const trainData = loadData(trainDataUrl);
        const ds = await tf.data
        .generator(dataGenerator)
        .batch(batchSize)
        await ds.forEachAsync(e => console.log());
        
        const saveModelPath = 'file://./model/model.json';
      
        const model = buildModel();
        model.summary();
      
        const info = await trainsModel(model, ds);
        console.log('\r\n', info);
        console.log('\r\nEvaluating model...');
        await evaluateModel(model, testData);
        console.log('\r\nSaving model...');
        await model.save(saveModelPath);
      };

      run()
      
    }

    


}