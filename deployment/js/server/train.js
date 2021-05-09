let numOfClasses = 6
let epochsvalue = 20
const batchSize = 100;
const imageWidth = 28;
const imageHeight = 28;
const imageChannels = 3;

// const trainDataUrl = 'file://./model/train.csv';


const genFeatureTensor = async imagePath => { 
    let image = await tf.browser.fromPixels(imagePath, 3)
    image = await tf.image.resizeBilinear(image, [imageWidth, imageHeight] )
    return image
  }  
  

  const labelArray = indice => Array.from({length: 1}, (_, k) => k === indice ? 1 : 0)
  
  async function* dataGenerator() {
    const numElements = 10;
    let index = 0;
    while (index < numElements) {
      const imagePath = imagess[index]
      const feature = await genFeatureTensor(imagePath) ;
      const label = tf.tensor1d(labelArray(labels))
      index++;
      yield {xs: feature, ys: label};
    }
  
  }
  
  // async function trainModel(){
  //   const ds = await tf.data
  //   .generator(dataGenerator)
  //   await ds.forEachAsync(e => console.log());
  //   // console.log(ds)
  //   const trainmodel = await tf.sequential();
  //     await trainmodel.add(tf.layers.conv2d({
  //       inputShape: [224, 224, 3], // numberOfChannels = 3 for colorful images and one otherwise
  //       filters: 32,
  //       kernelSize: 3,
  //       activation: 'relu',
  //     }));
  //     await trainmodel.add(tf.layers.flatten()),
  //     await trainmodel.add(tf.layers.dense({
  //       units: numOfClasses, 
  //       activation: 'softmax'
  //     }));
  //     await trainmodel.compile({
  //       optimizer: 'sgd',
  //       loss: 'meanSquaredError',
  //       // optimizer: 'adam',
  //       // loss: 'categoricalCrossentropy',
  //       metrics: ['accuracy']
  //     })
  //     await trainmodel.fitDataset(ds,  {
  //       epochs: epochsvalue,
  //       callbacks: {
  //         onEpochBegin: async (epoch, logs) => {
  //           console.log(`Epoch ${epoch + 1} of ${epochsvalue} ...`)
  //         },
  //         onEpochEnd: async (epoch, logs) => {
  //           console.log(`  train-set loss: ${logs.loss.toFixed(4)}`)
  //           console.log(`  train-set accuracy: ${logs.acc.toFixed(4)}`)
  //         }
  //       }
  //     });
  // }


  // const trainDataUrl = './model/fashion-mnist_test.csv';
  // // const testDataUrl = 'file://./fashion-mnist/fashion-mnist_test.csv';

  // const loadData = function (dataUrl, batches=batchSize) {
  //   // normalize data values between 0-1
  //   const normalize = ({xs, ys}) => {
  //     return {
  //         xs: Object.values(xs).map(x => x / 255),
  //         ys: ys.label
  //     };
  //   };
  
  //   // transform input array (xs) to 3D tensor
  //   // binarize output label (ys)
  //   const transform = ({xs, ys}) => {
  //     // array of zeros
  //     const zeros = (new Array(numOfClasses)).fill(0);
  
  //     return {
  //         xs: tf.tensor(xs, [imageWidth, imageHeight, imageChannels]),
  //         ys: tf.tensor1d(zeros.map((z, i) => {
  //             return i === ys ? 1 : 0;
  //         }))
  //     };
  //   };

  //   // load, normalize, transform, batch
  //   return tf.data
  //     .csv(dataUrl, {columnConfigs: {label: {isLabel: true}}})
  //     .map(normalize)
  //     .filter(f => f.ys < numOfClasses)
  //     .map(transform)
  //     .batch(batchSize);
  // };

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
    // console.log(flattenedDataset)
    // console.log(ds)
    // const testData = loadData(testDataUrl);
  
    // Full path to the directory to save the model in
    const saveModelPath = 'downloads://my-model';
  
    const model = buildModel();
    model.summary();
  
    const info = await trainsModel(model, ds);
    console.log('\r\n', info);
    console.log('\r\nEvaluating model...');
    // await evaluateModel(model, testData);
    console.log('\r\nSaving model...');
    await model.save(saveModelPath);
  };
  
 