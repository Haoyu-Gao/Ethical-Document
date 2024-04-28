---
language:
- pt
license: afl-3.0
datasets:
- squad
pipeline_tag: text2text-generation
widget:
- text: "extrair respostas: \nA Volkswagen anunciou a chegada do ID.Buzz, a Kombi\
    \ elétrica, ao Brasil. <hl> Em campanha publicitária, a marca alemã usou tecnologia\
    \ de inteligência artificial para criar um comercial com a cantora Elis Regina,\
    \ falecida em 1982, e a sua filha, a também cantora Maria Rita. <hl> Ambas aparecem\
    \ cantando juntas a música 'Como Nossos Pais', composta por Belchior e eternizada\
    \ por Elis. O vídeo, que já foi divulgado nas redes sociais da marca, foi exibido\
    \ pela primeira vez em comemoração de 70 anos da Volkswagen no ginásio do Ibirapuera,\
    \ em São Paulo."
  example_title: 1 - Extract Answers
- text: 'gerar pergunta: Em campanha publicitária, a marca alemã usou tecnologia de
    inteligência artificial para criar um comercial com a cantora Elis Regina, falecida
    em <hl> 1982 <hl>, e a sua filha, a também cantora Maria Rita.'
  example_title: 2 - Generate Questions ex-1
- text: 'gerar pergunta: Em campanha publicitária, a marca alemã usou tecnologia de
    inteligência artificial para criar um comercial com a cantora Elis Regina, falecida
    em 1982, e a sua filha, a também cantora <hl> Maria Rita <hl>.'
  example_title: 2 - Generate Questions ex-2
---
# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->
This model is intended to be used generating questions and answers from brazilian portuguese text passages, 
so you can finetune another BERT model into your generated triples (context-question-answer) for extractive question answering without supervision or labeled data.

It was trained using [unicamp-dl/ptt5-small-t5-portuguese-vocab](https://huggingface.co/unicamp-dl/ptt5-small-t5-portuguese-vocab) base model, [Squad 1.1 portuguese version](https://huggingface.co/datasets/ArthurBaia/squad_v1_pt_br) 
[Squad 2.0 portuguese version](https://github.com/cjaniake/squad_v2.0_pt) datasets to generante question and answers from text passages.

### Model Description

<!-- Provide a longer summary of what this model is. -->



- **Developed by:** Vitor Alcantara Batista (vabatista@gmail.com)
- **Model type:** T5 small
- **Language(s) (NLP):** Brazilian Portuguese
- **License:** [Academic Free License v. 3.0](https://opensource.org/license/afl-3-0-php/)
- **Finetuned from model :** unicamp-dl/ptt5-small-t5-portuguese-vocab

### Model Sources [optional]

<!-- Provide the basic links for the model. -->
  
- **Repository:** This model used code from this github repo [https://github.com/patil-suraj/question_generation/](https://github.com/patil-suraj/question_generation/)

## Usage

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

How to use it (after cloning the github repo above):

```
from pipelines import pipeline
nlp = pipeline("question-generation", model='vabatista/question-generation-t5-small-pt-br', tokenizer='vabatista/question-generation-t5-small-pt-br')

text = """ PUT YOUR TEXT PASSAGE HERE """
nlp(text) 

```
Sample usage/results:
```
text = """A Volkswagen anunciou a chegada do ID.Buzz, a Kombi elétrica, ao Brasil. Em campanha publicitária, a marca alemã usou tecnologia de inteligência artificial 
para criar um comercial com a cantora Elis Regina, falecida em 1982, e a sua filha, a também cantora Maria Rita. Ambas aparecem cantando juntas a música 'Como Nossos Pais', composta por Belchior e eternizada por Elis.
O vídeo, que já foi divulgado nas redes sociais da marca, foi exibido pela primeira vez em comemoração de 70 anos da Volkswagen no ginásio do Ibirapuera, em São Paulo.
Diante de 5 mil pessoas, entre funcionários e convidados, a apresentação ainda contou com a presença de Maria Rita, que também cantou ao vivo a canção e se emocionou bastante - 
a cantora chegou a chorar abraçada com Ciro Possobom, CEO da VW do Brasil.
A técnica utilizada, conhecida também como "deep fake", aplica IA para criar conteúdos realistas. No caso, foi produzida pela agência AlmapBBDO."""

nlp(text)

[{'answer': 'Kombi elétrica', 'question': 'Qual é o nome do ID.Buzz?'},
 {'answer': 'tecnologia de inteligência artificial',
  'question': 'O que a Volkswagen usou para criar um comercial com Elis Regina?'},
 {'answer': 'Como Nossos Pais',
  'question': 'Qual é o nome da música que Elis Regina cantou?'},
 {'answer': '70 anos',
  'question': 'Qual foi o aniversário da Volkswagen em comemoração ao ID.Buzz?'},
 {'answer': 'Ciro Possobom', 'question': 'Quem foi o CEO da VW do Brasil?'},
 {'answer': 'deep fake', 'question': 'Qual é o outro nome para o ID.Buzz?'},
 {'answer': 'AlmapBBDO', 'question': 'Qual agência produziu o ID.Buzz?'}]
```

You may also use this model directly using this inputs (you can test on the sandbox in this page):

1. extrair respostas: \<PHRASE HERE>

2. gerar pergunta: \<HIGHLIGHTED PHRASE HERE>
where \<HIGHLIGHTED PHRASE> uses \<hl> token to highlight generated answer.

Example:

input: "extrair respostas: A Volkswagen anunciou a chegada do ID.Buzz, a Kombi elétrica, ao Brasil."

output: ID.Buzz

input: "gerar perguntas: A Volkswagen anunciou a chegada do \<hl> ID.Buzz \<hl>, a Kombi elétrica, ao Brasil."

output: "Qual é o nome da Kombi elétrica da Volkswagen no Brasil?"




## Training Details

10 epochs, learning-rate 1e-4

## Model Card Authors

Vitor Alcantara Batista

## Model Card Contact

vabatista@gmail.com