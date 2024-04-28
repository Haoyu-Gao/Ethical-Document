---
language:
- eng
license: apache-2.0
tags:
- text-classification
- Sentiment
- RoBERTa
- Financial Statements
- Accounting
- Finance
- Business
- ESG
- CSR Reports
- Financial News
- Earnings Call Transcripts
- Sustainability
- Corporate governance
---
<!DOCTYPE html>
<html>
<body>

<h1><b>Financial-RoBERTa</b></h1>
<p><b>Financial-RoBERTa</b> is a pre-trained NLP model to analyze sentiment of financial text including:</p>
<ul style="PADDING-LEFT: 40px">
  <li>Financial Statements,</li>
  <li>Earnings Announcements,</li>
  <li>Earnings Call Transcripts,</li>
  <li>Corporate Social Responsibility (CSR) Reports,</li>
  <li>Environmental, Social, and Governance (ESG) News,</li>
  <li>Financial News,</li>
  <li>Etc.</li>
</ul>
<p>Financial-RoBERTa is built by further training and fine-tuning the RoBERTa Large language model using a large corpus created from 10k, 10Q, 8K, Earnings Call Transcripts, CSR Reports, ESG News, and Financial News text.</p>
<p>The model will give softmax outputs for three labels: <b>Positive</b>, <b>Negative</b> or <b>Neutral</b>.</p>
<p><b>How to perform sentiment analysis:</b></p>
<p>The easiest way to use the model for single predictions is Hugging Face's sentiment analysis pipeline, which only needs a couple lines of code as shown in the following example:</p>
<pre>
  <code>
from transformers import pipeline
sentiment_analysis = pipeline("sentiment-analysis",model="soleimanian/financial-roberta-large-sentiment")
print(sentiment_analysis("In fiscal 2021, we generated a net yield of approximately 4.19% on our investments, compared to approximately 5.10% in fiscal 2020."))
  </code>
</pre>
<p>I provide an example script via <a href="https://colab.research.google.com/drive/11RGWU3UDtxnjan8Ug6dyX82m9fBV6CGo?usp=sharing" target="_blank">Google Colab</a>. You can load your data to a Google Drive and run the script for free on a Colab.
<p><b>Citation and contact:</b></p>
<p>Please cite <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4115943" target="_blank">this paper</a> when you use the model. Feel free to reach out to mohammad.soleimanian@concordia.ca with any questions or feedback you may have.<p/>

</body>
</html>
