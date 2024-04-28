---
language:
- multilingual
license: cc-by-nc-sa-4.0
library_name: span-marker
tags:
- span-marker
- token-classification
- ner
- named-entity-recognition
datasets:
- Babelscape/multinerd
metrics:
- f1
- recall
- precision
pipeline_tag: token-classification
widget:
- text: Amelia Earthart flog mit ihrer einmotorigen Lockheed Vega 5B über den Atlantik
    nach Paris.
  example_title: German
- text: Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic
    to Paris.
  example_title: English
- text: Amelia Earthart voló su Lockheed Vega 5B monomotor a través del Océano Atlántico
    hasta París.
  example_title: Spanish
- text: Amelia Earthart a fait voler son monomoteur Lockheed Vega 5B à travers l'ocean
    Atlantique jusqu'à Paris.
  example_title: French
- text: Amelia Earhart ha volato con il suo monomotore Lockheed Vega 5B attraverso
    l'Atlantico fino a Parigi.
  example_title: Italian
- text: Amelia Earthart vloog met haar één-motorige Lockheed Vega 5B over de Atlantische
    Oceaan naar Parijs.
  example_title: Dutch
- text: Amelia Earthart przeleciała swoim jednosilnikowym samolotem Lockheed Vega
    5B przez Ocean Atlantycki do Paryża.
  example_title: Polish
- text: Amelia Earhart voou em seu monomotor Lockheed Vega 5B através do Atlântico
    para Paris.
  example_title: Portuguese
- text: Амелия Эртхарт перелетела на своем одномоторном самолете Lockheed Vega 5B
    через Атлантический океан в Париж.
  example_title: Russian
- text: Amelia Earthart flaug eins hreyfils Lockheed Vega 5B yfir Atlantshafið til
    Parísar.
  example_title: Icelandic
- text: Η Amelia Earthart πέταξε το μονοκινητήριο Lockheed Vega 5B της πέρα ​​από
    τον Ατλαντικό Ωκεανό στο Παρίσι.
  example_title: Greek
- text: Amelia Earhartová přeletěla se svým jednomotorovým Lockheed Vega 5B přes Atlantik
    do Paříže.
  example_title: Czech
- text: Amelia Earhart lensi yksimoottorisella Lockheed Vega 5B:llä Atlantin yli Pariisiin.
  example_title: Finnish
- text: Amelia Earhart fløj med sin enmotoriske Lockheed Vega 5B over Atlanten til
    Paris.
  example_title: Danish
- text: Amelia Earhart flög sin enmotoriga Lockheed Vega 5B över Atlanten till Paris.
  example_title: Swedish
- text: Amelia Earhart fløy sin enmotoriske Lockheed Vega 5B over Atlanterhavet til
    Paris.
  example_title: Norwegian
- text: Amelia Earhart și-a zburat cu un singur motor Lockheed Vega 5B peste Atlantic
    până la Paris.
  example_title: Romanian
- text: Amelia Earhart menerbangkan mesin tunggal Lockheed Vega 5B melintasi Atlantik
    ke Paris.
  example_title: Indonesian
- text: Амелія Эрхарт пераляцела на сваім аднаматорным Lockheed Vega 5B праз Атлантыку
    ў Парыж.
  example_title: Belarusian
- text: Амелія Ергарт перелетіла на своєму одномоторному літаку Lockheed Vega 5B через
    Атлантику до Парижа.
  example_title: Ukrainian
- text: Amelia Earhart preletjela je svojim jednomotornim zrakoplovom Lockheed Vega
    5B preko Atlantika do Pariza.
  example_title: Croatian
- text: Amelia Earhart lendas oma ühemootoriga Lockheed Vega 5B üle Atlandi ookeani
    Pariisi .
  example_title: Estonian
base_model: bert-base-multilingual-cased
model-index:
- name: SpanMarker w. bert-base-multilingual-cased on MultiNERD by Tom Aarsen
  results:
  - task:
      type: token-classification
      name: Named Entity Recognition
    dataset:
      name: MultiNERD
      type: Babelscape/multinerd
      split: test
      revision: 2814b78e7af4b5a1f1886fe7ad49632de4d9dd25
    metrics:
    - type: f1
      value: 0.92478
      name: F1
    - type: precision
      value: 0.93385
      name: Precision
    - type: recall
      value: 0.91588
      name: Recall
---

# SpanMarker for Multilingual Named Entity Recognition

This is a [SpanMarker](https://github.com/tomaarsen/SpanMarkerNER) model that can be used for multilingual Named Entity Recognition trained on the [MultiNERD](https://huggingface.co/datasets/Babelscape/multinerd) dataset. In particular, this SpanMarker model uses [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) as the underlying encoder. See [train.py](train.py) for the training script.

Is your data not (always) capitalized correctly? Then consider using this uncased variant of this model by [@lxyuan](https://huggingface.co/lxyuan) for better performance: 
[lxyuan/span-marker-bert-base-multilingual-uncased-multinerd](https://huggingface.co/lxyuan/span-marker-bert-base-multilingual-uncased-multinerd).

## Metrics

| **Language** | **Precision** | **Recall** | **F1**     |
|--------------|---------------|------------|------------|
| **all**      | 93.39         | 91.59      | **92.48**  |
| **de**       | 95.21         | 94.32      | **94.76**  |
| **en**       | 95.07         | 95.29      | **95.18**  |
| **es**       | 93.50         | 89.65      | **91.53**  |
| **fr**       | 93.86         | 90.07      | **91.92**  |
| **it**       | 91.63         | 93.57      | **92.59**  |
| **nl**       | 94.86         | 91.74      | **93.27**  |
| **pl**       | 93.51         | 91.83      | **92.66**  |
| **pt**       | 94.48         | 91.30      | **92.86**  |
| **ru**       | 93.70         | 93.10      | **93.39**  |
| **zh**       | 88.36         | 85.71      | **87.02**  |

## Label set

| Class | Description | Examples |
|-------|-------------|----------|
PER (person) | People | Ray Charles, Jessica Alba, Leonardo DiCaprio, Roger Federer, Anna Massey. |
ORG (organization) | Associations, companies, agencies, institutions, nationalities and religious or political groups | University of Edinburgh, San Francisco Giants, Google, Democratic Party. |
LOC (location) | Physical locations (e.g. mountains, bodies of water), geopolitical entities (e.g. cities, states), and facilities (e.g. bridges, buildings, airports). | Rome, Lake Paiku, Chrysler Building, Mount Rushmore, Mississippi River. |
ANIM (animal) | Breeds of dogs, cats and other animals, including their scientific names. | Maine Coon, African Wild Dog, Great White Shark, New Zealand Bellbird. |
BIO (biological) | Genus of fungus, bacteria and protoctists, families of viruses, and other biological entities. | Herpes Simplex Virus, Escherichia Coli, Salmonella, Bacillus Anthracis. |
CEL (celestial) | Planets, stars, asteroids, comets, nebulae, galaxies and other astronomical objects. | Sun, Neptune, Asteroid 187 Lamberta, Proxima Centauri, V838 Monocerotis. |
DIS (disease) | Physical, mental, infectious, non-infectious, deficiency, inherited, degenerative, social and self-inflicted diseases. | Alzheimer’s Disease, Cystic Fibrosis, Dilated Cardiomyopathy, Arthritis. |
EVE (event) | Sport events, battles, wars and other events. | American Civil War, 2003 Wimbledon Championships, Cannes Film Festival. |
FOOD (food) | Foods and drinks. | Carbonara, Sangiovese, Cheddar Beer Fondue, Pizza Margherita. |
INST (instrument) | Technological instruments, mechanical instruments, musical instruments, and other tools. | Spitzer Space Telescope, Commodore 64, Skype, Apple Watch, Fender Stratocaster. |
MEDIA (media) | Titles of films, books, magazines, songs and albums, fictional characters and languages. | Forbes, American Psycho, Kiss Me Once, Twin Peaks, Disney Adventures. |
PLANT (plant) | Types of trees, flowers, and other plants, including their scientific names. | Salix, Quercus Petraea, Douglas Fir, Forsythia, Artemisia Maritima. |
MYTH (mythological) | Mythological and religious entities. | Apollo, Persephone, Aphrodite, Saint Peter, Pope Gregory I, Hercules. |
TIME (time) | Specific and well-defined time intervals, such as eras, historical periods, centuries, years and important days. No months and days of the week. | Renaissance, Middle Ages, Christmas, Great Depression, 17th Century, 2012. |
VEHI (vehicle) | Cars, motorcycles and other vehicles. | Ferrari Testarossa, Suzuki Jimny, Honda CR-X, Boeing 747, Fairey Fulmar.

## Usage

To use this model for inference, first install the `span_marker` library:

```bash
pip install span_marker
```

You can then run inference with this model like so:

```python
from span_marker import SpanMarkerModel

# Download from the 🤗 Hub
model = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-mbert-base-multinerd")
# Run inference
entities = model.predict("Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris.")
```

See the [SpanMarker](https://github.com/tomaarsen/SpanMarkerNER) repository for documentation and additional information on this library.

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 1

### Training results

| Training Loss | Epoch | Step   | Validation Loss | Overall Precision | Overall Recall | Overall F1 | Overall Accuracy |
|:-------------:|:-----:|:------:|:---------------:|:-----------------:|:--------------:|:----------:|:----------------:|
| 0.0179        | 0.01  | 1000   | 0.0146          | 0.8101            | 0.7616         | 0.7851     | 0.9530           |
| 0.0099        | 0.02  | 2000   | 0.0091          | 0.8571            | 0.8425         | 0.8498     | 0.9663           |
| 0.0085        | 0.03  | 3000   | 0.0078          | 0.8729            | 0.8579         | 0.8653     | 0.9700           |
| 0.0075        | 0.04  | 4000   | 0.0072          | 0.8821            | 0.8724         | 0.8772     | 0.9739           |
| 0.0074        | 0.05  | 5000   | 0.0075          | 0.8622            | 0.8841         | 0.8730     | 0.9722           |
| 0.0074        | 0.06  | 6000   | 0.0067          | 0.9056            | 0.8568         | 0.8805     | 0.9749           |
| 0.0066        | 0.07  | 7000   | 0.0065          | 0.9082            | 0.8543         | 0.8804     | 0.9737           |
| 0.0063        | 0.08  | 8000   | 0.0066          | 0.9039            | 0.8617         | 0.8823     | 0.9745           |
| 0.0062        | 0.09  | 9000   | 0.0062          | 0.9323            | 0.8425         | 0.8852     | 0.9754           |
| 0.007         | 0.1   | 10000  | 0.0066          | 0.8898            | 0.8758         | 0.8827     | 0.9746           |
| 0.006         | 0.11  | 11000  | 0.0061          | 0.8986            | 0.8841         | 0.8913     | 0.9766           |
| 0.006         | 0.12  | 12000  | 0.0061          | 0.9171            | 0.8628         | 0.8891     | 0.9763           |
| 0.0062        | 0.13  | 13000  | 0.0060          | 0.9264            | 0.8634         | 0.8938     | 0.9772           |
| 0.0059        | 0.14  | 14000  | 0.0059          | 0.9323            | 0.8508         | 0.8897     | 0.9763           |
| 0.0059        | 0.15  | 15000  | 0.0060          | 0.9011            | 0.8815         | 0.8912     | 0.9758           |
| 0.0059        | 0.16  | 16000  | 0.0060          | 0.9221            | 0.8598         | 0.8898     | 0.9763           |
| 0.0056        | 0.17  | 17000  | 0.0058          | 0.9098            | 0.8839         | 0.8967     | 0.9775           |
| 0.0055        | 0.18  | 18000  | 0.0060          | 0.9103            | 0.8739         | 0.8917     | 0.9765           |
| 0.0054        | 0.19  | 19000  | 0.0056          | 0.9135            | 0.8726         | 0.8925     | 0.9774           |
| 0.0052        | 0.2   | 20000  | 0.0058          | 0.9108            | 0.8834         | 0.8969     | 0.9773           |
| 0.0053        | 0.21  | 21000  | 0.0058          | 0.9038            | 0.8866         | 0.8951     | 0.9773           |
| 0.0057        | 0.22  | 22000  | 0.0057          | 0.9130            | 0.8762         | 0.8942     | 0.9775           |
| 0.0056        | 0.23  | 23000  | 0.0053          | 0.9375            | 0.8604         | 0.8973     | 0.9781           |
| 0.005         | 0.24  | 24000  | 0.0054          | 0.9253            | 0.8822         | 0.9032     | 0.9784           |
| 0.0055        | 0.25  | 25000  | 0.0055          | 0.9182            | 0.8807         | 0.8991     | 0.9787           |
| 0.0049        | 0.26  | 26000  | 0.0053          | 0.9311            | 0.8702         | 0.8997     | 0.9783           |
| 0.0051        | 0.27  | 27000  | 0.0054          | 0.9192            | 0.8877         | 0.9032     | 0.9787           |
| 0.0051        | 0.28  | 28000  | 0.0053          | 0.9332            | 0.8783         | 0.9049     | 0.9795           |
| 0.0049        | 0.29  | 29000  | 0.0054          | 0.9311            | 0.8672         | 0.8981     | 0.9789           |
| 0.0047        | 0.3   | 30000  | 0.0054          | 0.9165            | 0.8954         | 0.9058     | 0.9796           |
| 0.005         | 0.31  | 31000  | 0.0052          | 0.9079            | 0.9016         | 0.9047     | 0.9787           |
| 0.0051        | 0.32  | 32000  | 0.0051          | 0.9157            | 0.9001         | 0.9078     | 0.9796           |
| 0.0046        | 0.33  | 33000  | 0.0051          | 0.9147            | 0.8935         | 0.9040     | 0.9788           |
| 0.0046        | 0.34  | 34000  | 0.0050          | 0.9229            | 0.8847         | 0.9034     | 0.9793           |
| 0.005         | 0.35  | 35000  | 0.0051          | 0.9198            | 0.8922         | 0.9058     | 0.9796           |
| 0.0047        | 0.36  | 36000  | 0.0050          | 0.9321            | 0.8890         | 0.9100     | 0.9807           |
| 0.0048        | 0.37  | 37000  | 0.0050          | 0.9046            | 0.9133         | 0.9089     | 0.9800           |
| 0.0046        | 0.38  | 38000  | 0.0051          | 0.9170            | 0.8973         | 0.9071     | 0.9806           |
| 0.0048        | 0.39  | 39000  | 0.0050          | 0.9417            | 0.8775         | 0.9084     | 0.9805           |
| 0.0042        | 0.4   | 40000  | 0.0049          | 0.9238            | 0.8937         | 0.9085     | 0.9797           |
| 0.0038        | 0.41  | 41000  | 0.0048          | 0.9371            | 0.8920         | 0.9140     | 0.9812           |
| 0.0042        | 0.42  | 42000  | 0.0048          | 0.9359            | 0.8862         | 0.9104     | 0.9808           |
| 0.0051        | 0.43  | 43000  | 0.0049          | 0.9080            | 0.9060         | 0.9070     | 0.9805           |
| 0.0037        | 0.44  | 44000  | 0.0049          | 0.9328            | 0.8877         | 0.9097     | 0.9801           |
| 0.0041        | 0.45  | 45000  | 0.0049          | 0.9231            | 0.8975         | 0.9101     | 0.9813           |
| 0.0046        | 0.46  | 46000  | 0.0046          | 0.9308            | 0.8943         | 0.9122     | 0.9812           |
| 0.0038        | 0.47  | 47000  | 0.0047          | 0.9291            | 0.8969         | 0.9127     | 0.9815           |
| 0.0043        | 0.48  | 48000  | 0.0046          | 0.9308            | 0.8909         | 0.9104     | 0.9804           |
| 0.0043        | 0.49  | 49000  | 0.0046          | 0.9278            | 0.8954         | 0.9113     | 0.9800           |
| 0.0039        | 0.5   | 50000  | 0.0047          | 0.9173            | 0.9073         | 0.9123     | 0.9817           |
| 0.0043        | 0.51  | 51000  | 0.0045          | 0.9347            | 0.8962         | 0.9150     | 0.9821           |
| 0.0047        | 0.52  | 52000  | 0.0045          | 0.9266            | 0.9016         | 0.9139     | 0.9810           |
| 0.0035        | 0.53  | 53000  | 0.0046          | 0.9165            | 0.9122         | 0.9144     | 0.9820           |
| 0.0038        | 0.54  | 54000  | 0.0046          | 0.9231            | 0.9050         | 0.9139     | 0.9823           |
| 0.0036        | 0.55  | 55000  | 0.0046          | 0.9331            | 0.9005         | 0.9165     | 0.9828           |
| 0.0037        | 0.56  | 56000  | 0.0047          | 0.9246            | 0.9016         | 0.9129     | 0.9821           |
| 0.0035        | 0.57  | 57000  | 0.0044          | 0.9351            | 0.9003         | 0.9174     | 0.9829           |
| 0.0043        | 0.57  | 58000  | 0.0043          | 0.9257            | 0.9079         | 0.9167     | 0.9826           |
| 0.004         | 0.58  | 59000  | 0.0043          | 0.9286            | 0.9065         | 0.9174     | 0.9823           |
| 0.0041        | 0.59  | 60000  | 0.0044          | 0.9324            | 0.9050         | 0.9185     | 0.9825           |
| 0.0039        | 0.6   | 61000  | 0.0044          | 0.9268            | 0.9041         | 0.9153     | 0.9815           |
| 0.0038        | 0.61  | 62000  | 0.0043          | 0.9367            | 0.8918         | 0.9137     | 0.9819           |
| 0.0037        | 0.62  | 63000  | 0.0044          | 0.9249            | 0.9160         | 0.9205     | 0.9833           |
| 0.0036        | 0.63  | 64000  | 0.0043          | 0.9398            | 0.8975         | 0.9181     | 0.9827           |
| 0.0036        | 0.64  | 65000  | 0.0043          | 0.9260            | 0.9118         | 0.9188     | 0.9829           |
| 0.0035        | 0.65  | 66000  | 0.0044          | 0.9375            | 0.8988         | 0.9178     | 0.9828           |
| 0.0034        | 0.66  | 67000  | 0.0043          | 0.9272            | 0.9143         | 0.9207     | 0.9833           |
| 0.0033        | 0.67  | 68000  | 0.0044          | 0.9332            | 0.9024         | 0.9176     | 0.9827           |
| 0.0035        | 0.68  | 69000  | 0.0044          | 0.9396            | 0.8981         | 0.9184     | 0.9825           |
| 0.0038        | 0.69  | 70000  | 0.0042          | 0.9265            | 0.9163         | 0.9214     | 0.9827           |
| 0.0035        | 0.7   | 71000  | 0.0044          | 0.9375            | 0.9013         | 0.9191     | 0.9827           |
| 0.0037        | 0.71  | 72000  | 0.0042          | 0.9264            | 0.9171         | 0.9217     | 0.9830           |
| 0.0039        | 0.72  | 73000  | 0.0043          | 0.9399            | 0.9003         | 0.9197     | 0.9826           |
| 0.0039        | 0.73  | 74000  | 0.0041          | 0.9341            | 0.9094         | 0.9216     | 0.9832           |
| 0.0035        | 0.74  | 75000  | 0.0042          | 0.9301            | 0.9160         | 0.9230     | 0.9837           |
| 0.0037        | 0.75  | 76000  | 0.0042          | 0.9342            | 0.9107         | 0.9223     | 0.9835           |
| 0.0034        | 0.76  | 77000  | 0.0042          | 0.9331            | 0.9118         | 0.9223     | 0.9836           |
| 0.003         | 0.77  | 78000  | 0.0041          | 0.9330            | 0.9135         | 0.9231     | 0.9838           |
| 0.0034        | 0.78  | 79000  | 0.0041          | 0.9308            | 0.9082         | 0.9193     | 0.9832           |
| 0.0037        | 0.79  | 80000  | 0.0040          | 0.9346            | 0.9128         | 0.9236     | 0.9839           |
| 0.0032        | 0.8   | 81000  | 0.0041          | 0.9389            | 0.9128         | 0.9257     | 0.9841           |
| 0.0031        | 0.81  | 82000  | 0.0040          | 0.9293            | 0.9163         | 0.9227     | 0.9836           |
| 0.0032        | 0.82  | 83000  | 0.0041          | 0.9305            | 0.9160         | 0.9232     | 0.9835           |
| 0.0034        | 0.83  | 84000  | 0.0041          | 0.9327            | 0.9118         | 0.9221     | 0.9838           |
| 0.0028        | 0.84  | 85000  | 0.0041          | 0.9279            | 0.9216         | 0.9247     | 0.9839           |
| 0.0031        | 0.85  | 86000  | 0.0041          | 0.9326            | 0.9167         | 0.9246     | 0.9838           |
| 0.0029        | 0.86  | 87000  | 0.0040          | 0.9354            | 0.9158         | 0.9255     | 0.9841           |
| 0.0031        | 0.87  | 88000  | 0.0041          | 0.9327            | 0.9156         | 0.9241     | 0.9840           |
| 0.0033        | 0.88  | 89000  | 0.0040          | 0.9367            | 0.9141         | 0.9253     | 0.9846           |
| 0.0031        | 0.89  | 90000  | 0.0040          | 0.9379            | 0.9141         | 0.9259     | 0.9844           |
| 0.0031        | 0.9   | 91000  | 0.0040          | 0.9297            | 0.9184         | 0.9240     | 0.9843           |
| 0.0034        | 0.91  | 92000  | 0.0040          | 0.9299            | 0.9188         | 0.9243     | 0.9843           |
| 0.0036        | 0.92  | 93000  | 0.0039          | 0.9324            | 0.9175         | 0.9249     | 0.9843           |
| 0.0028        | 0.93  | 94000  | 0.0039          | 0.9399            | 0.9135         | 0.9265     | 0.9848           |
| 0.0029        | 0.94  | 95000  | 0.0040          | 0.9342            | 0.9173         | 0.9257     | 0.9845           |
| 0.003         | 0.95  | 96000  | 0.0040          | 0.9378            | 0.9184         | 0.9280     | 0.9850           |
| 0.0029        | 0.96  | 97000  | 0.0039          | 0.9380            | 0.9152         | 0.9264     | 0.9847           |
| 0.003         | 0.97  | 98000  | 0.0039          | 0.9372            | 0.9156         | 0.9263     | 0.9849           |
| 0.003         | 0.98  | 99000  | 0.0039          | 0.9387            | 0.9167         | 0.9276     | 0.9851           |
| 0.0031        | 0.99  | 100000 | 0.0039          | 0.9373            | 0.9177         | 0.9274     | 0.9849           |

### Framework versions

- SpanMarker 1.2.4
- Transformers 4.28.1
- Pytorch 1.13.1+cu117
- Datasets 2.12.0
- Tokenizers 0.13.2

## See also
* [lxyuan/span-marker-bert-base-multilingual-cased-multinerd](https://huggingface.co/lxyuan/span-marker-bert-base-multilingual-cased-multinerd) is similar to this model, but trained on 3 epochs instead of 2. It reaches better performance on 7 out of the 10 languages.
* [lxyuan/span-marker-bert-base-multilingual-uncased-multinerd](https://huggingface.co/lxyuan/span-marker-bert-base-multilingual-uncased-multinerd) is a strong uncased variant of this model, also trained on 3 epochs instead of 2.

## Contributions
Many thanks to [Simone Tedeschi](https://huggingface.co/sted97) from [Babelscape](https://babelscape.com) for his insight when training this model and his involvement in the creation of the training dataset.
