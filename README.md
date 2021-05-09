# eindproject-rikvermeulen
eindproject-rikvermeulen created by GitHub Classroom

# Concept
Mijn concept voor de eindopdracht PRG 8 is “Trashtalk”. Dit is een machine die detecteert wat voor afval het is door er een foto van te nemen en dit te analyseren. Dit idee komt oorspronkelijk uit de eerste weken van periode tijdens de design sprint. Dit concept moet ervoor zorgen dat het afval correct gescheiden wordt. Dit maakt het voor de gebruiker gemakkelijker om te bepalen wat voor soort afval het is. Machine learning helpt hierbij omdat je dit zo kan trainen dat hij bijna elk afval kan herkennen.
Zelf had ik al wat onderzoek gedaan naar verschillende libraries, tools en API’s om het idee te realiseren. Ik heb uit eindelijk besloten om de basis van het idee te realiseren met een Raspberry Pi (3 model b). Dit geeft mij de mogelijkheid het proces te starten met een fysieke knop, met de camera module live een foto van het object te nemen en te analyseren zodat het resultaat aan wordt geven met LED’s. Dit betekende dat ik libraries nodig had om deze modules aan te sturen. Na wat onderzoek kwam ik op twee programmeer talen uit wat mij zou kunnen helpen in het realiseren van dit concept. Javascript met nadruk op het platform nodejs (Prototype1) of Python (Prototype2). 
Het was ook van belang om te onderzoeken welke libraries ik ging gebruiken voor machine learning. Ik kwam al gauw uit bij Tensorflow omdat ze een brede documentatie hebben voor beide talen en over de raspberry.

# Prototype 1
Mijn eerste prototype is geschreven in Javascript in de Nodejs omgeving. Dit prototype had als basis mogelijkheid van het detecteren van afval door middel van de camera modulen. Dit script werd aangestuurd door tensorflow. Het model is getrained in de teachable machine en ik gebruikte verschillende libraries uit nodejs doormiddel van “npm” (zie alle modules in “package.json”). Ik gebruikte training data van een dataset uit kaggle “”. Zelf gebruik ik dan ook classification in beide prototypes om een object te herkennen. De basis van dit prototype werkt lokaal op de Raspberry en kan zonder enige inmenging werken. Ik ben daarbij wel van plan een adminpaneel te ontwikkelen in  een web app.

## Resultaat:
Dit prototype resulteerde in de volgende data als ik een afbeelding van papier liet zien:
![alt text](https://github.com/HR-CMGT-Classroom/eindproject-rikvermeulen/blob/master/development/images/github/predictPrototype1.PNG?raw=true)<br>

# Prototype 2
Het tweede prototype heb ik ontwikkeld in Python. Dit was voor mij een stuk lastiger want deze taal is compleet nieuw voor mij. Het is met uiteindelijk gelukt een basis neer te zetten waarbij alle functionaliteiten werken. Voor dit prototype gebruikt ik module van “lobe” om het model te maken en classificeren van de objecten. Ik miste hierbij  de vrijheid van tensorflow een beetje. Dit had misschien ook te maken dat python onbekend aan mij is. De resultaten uit dit algoritme kwamen wel veelal overheen met prototype 2. Er was ook veel minder code nodig in vergelijking met javascript om de basis werkend te krijgen.

## Resultaat:
Dit prototype resulteerde in de volgende data als ik een afbeelding van papier liet zien:
![alt text](https://github.com/HR-CMGT-Classroom/eindproject-rikvermeulen/blob/master/development/images/github/predictPrototype2.PNG?raw=true)<br>

# Conclusie
Ik heb besloten om verder te gaan met prototype 1. Dit omdat ik veel kennis heb van javascript, Nodejs en Tensorflow. Ik weet hierin de weg makkelijker te vinden. Uit het testen blijkt dan ook dat Prototype beter scored qua accurate in het bepalen van afval in vergelijking met Prototype 2. Zie Resultaten hierboven. Ik weet dat ik de score van Prototype 2 zou kunnen verbeteren alleen dit kost mij veel moeite omdat ik daarbij ook een nieuwe taal moet bestuderen in een korte tijd. Zelf heb ik wel onderzoek gedaan naar projecten die ik in de toekomst met python kan coderen.


# Uiwerking en web Aplicatie
## Functionaliteiten
Nadat ik had besloten welk prototype ik verder ging uitwerken ging ik aan de slag met de nieuwe functionaliteiten, een admin panel draaiend via express en train functionaliteit om het model te trainen en op te slaan. Via het admin paneel is het mogelijk een foto te uploaden via de browser en het model te laten bepalen wat voor afval dit is. Ook is het mogelijk om een model te trainen door middel van foto’s te uploaden. Dit is ook mogelijk op de Raspberry pi alleen door de lage RAM en rekenkracht heeft dit weinig zin. De Pi kan geen hoge epoch en batch size aan waardoor het meestal een training is van niks. Daarom heb ik ervoor gekozen deze functionaliteit in de web interface te bouwen. Dit kan gebruik maken van de CPU en GPU van de computer waar het in geopend is. De web interface heb duidelijk ingedeeld en gestyled met css en javascript om het de gebruiker makkelijk. Het geeft duidelijk aan als er iets mis gaat of als er nog iets mist.

![alt text](https://github.com/HR-CMGT-Classroom/eindproject-rikvermeulen/blob/master/development/images/github/interface.PNG?raw=true)<br>

## Raspberry Pi prototype
De Raspberry Pi heb ik zo in elkaar gezet dat je het kan bedienen met 1 knop. Als de knop wordt ingedrukt start het proces, gaat het witten licht aan wat geld als een flits en maakt de camera een foto. Deze foto wordt opgeslagen in de directory en wordt daarna doorgestuurde naar de functie die moet bepalen wat het gaat worden. Nadat het resultaat bekend is wordt dit aangeven met LED’s aanwezig op het breadbord. Elk Led lichtje staat voor een label waarbij het LED aan gaat als dit overheen komt met het resultaat van het model.

![alt text](https://github.com/HR-CMGT-Classroom/eindproject-rikvermeulen/blob/master/development/images/github/concept.jpg?raw=true)<br>

### Video:
[Youtube video Raspi voorbeeld 1](https://youtu.be/GGgpdt_vFMY)<br>
[Youtube video Raspi voorbeeld 2](https://youtu.be/5jEjgmylffY)<br>

# Getting starded

# Hardware
- Raspberry Pi 3 Model B of nieuwer
[Link](https://www.adafruit.com/product/4292)
- Raspberry Pi Camera Board v2 - 8 Megapixels
[Link](https://www.adafruit.com/product/3099)
- Power Supply 5.1V 3A met Micro USB of USB C (Raspi 4)
[Link](https://www.adafruit.com/product/4298)
- Half-size breadboard
[Link](https://www.adafruit.com/product/64)
- 1 x Pushbutton
- 6 x LEDs (Meedere kleuren)
- 5 x 220 Ohm Resistors
- 8 x Jumper wires

![alt text](https://github.com/HR-CMGT-Classroom/eindproject-rikvermeulen/blob/master/development/images/github/blueprint.PNG?raw=true)

# Software (PC)
- Teachable machine[Teachable machine](https://lobe.ai/)<br>
- Lobe (prototype 2)[Lobe](https://teachablemachine.withgoogle.com/train/image)<br>
- Terminal<br>

# Install
Houd er rekening mee dat dit alles draait op een linux based system. Dit betekend dat je tegen problemen kan aanlopen als het probeert op een andere OS zoals Windows of MacOS. Bijvoorbeeld python kan je niet zomaar draaien op een OS zonder installatie van Python zelf.

## Prototype(1)
Clone de repository<br>
```git clone https://github.com/HR-CMGT-Classroom/eindproject-rikvermeulen/deployment.git```<br />
`cd` in de root map<br>
Installeer npm<br>
```npm i```<br />
installeren van nodetf<br>
```sudo npm install @tensorflow/tfjs-node```<br>
Start de applicatie<br>
```node app.js```

### Custom model
Maak een model op [Teachable machine](https://teachablemachine.withgoogle.com/train/image)<br>
Verplaats de files na het exporten in de map `./model`

### Express
Admin paneel Trashtalk
stel uw express root locatie in door `dir` aan te passen in `server.js`, standaard is `./`<br />
Stel uw localhost port in bij `app.listen` in `server,js`, standaard is `8080`<br />
Start express met `node app.js` of `server.js`

## Prototype 2
Clone de repository<br>
```git clone https://github.com/HR-CMGT-Classroom/eindproject-rikvermeulen/Prototype2.git```<br />
`cd` in de root map<br>
Lobe module installeren<br> 
`pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl`<br>
`pip3 install lobe`<br>
Start applicatie<br>
`sudo python3 classifier`<br>

### Custom model
Installeer [Lobe](https://lobe.ai/)<br>
Verplaats de files na het exporten in de map `./model`

# License

MIT

**Free Software, Hell Yeah!**
