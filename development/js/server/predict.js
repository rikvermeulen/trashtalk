
async function classify(pre_image) {
   
    pre_image = tf.expandDims(pre_image, axis=-0)
    pre_image = tf.image.cropAndResize(image=pre_image, boxes=[[.1,.1,.6,.6]], [0], crop_size=[224, 224])
  
    const model = await tf.loadLayersModel(handler);
    const logits = tf.tidy(() => {
      return model.predict(pre_image);
    });
  
    const classes = await getTopKClasses(labels, logits, 3);
    const topResult = classes[0].className;
    console.log('My prediction is:', topResult);
    document.getElementById("predictions").innerHTML = topResult
    speak('My prediction is:' + topResult)
    document.getElementById("probability").innerHTML = classes[0].probability
    
    console.log('All predicitons are:', classes); 
  }
  
  async function getTopKClasses(labels, logits, topK = 3) {
    const values = await logits.data();
  
    return tf.tidy(() => {
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