const tfnode = require('@tensorflow/tfjs-node');

module.exports = {
    TopClasses: async function getTopKClasses(labels, logits, topK = 3) {
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
}