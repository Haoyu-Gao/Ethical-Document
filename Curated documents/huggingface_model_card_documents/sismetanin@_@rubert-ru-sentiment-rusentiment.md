---
language:
- ru
tags:
- sentiment analysis
- Russian
---

## RuBERT-Base-ru-sentiment-RuSentiment
RuBERT-ru-sentiment-RuSentiment is a [RuBERT](https://huggingface.co/DeepPavlov/rubert-base-cased) model fine-tuned on [RuSentiment dataset](https://github.com/text-machine-lab/rusentiment) of general-domain Russian-language posts from the largest Russian social network, VKontakte. 
<table>
<thead>
  <tr>
    <th rowspan="4">Model</th>
    <th rowspan="4">Score<br></th>
    <th rowspan="4">Rank</th>
    <th colspan="12">Dataset</th>
  </tr>
  <tr>
    <td colspan="6">SentiRuEval-2016<br></td>
    <td colspan="2" rowspan="2">RuSentiment</td>
    <td rowspan="2">KRND</td>
    <td rowspan="2">LINIS Crowd</td>
    <td rowspan="2">RuTweetCorp</td>
    <td rowspan="2">RuReviews</td>
  </tr>
  <tr>
    <td colspan="3">TC</td>
    <td colspan="3">Banks</td>
  </tr>
  <tr>
    <td>micro F1</td>
    <td>macro F1</td>
    <td>F1</td>
    <td>micro F1</td>
    <td>macro F1</td>
    <td>F1</td>
    <td>wighted</td>
    <td>F1</td>
    <td>F1</td>
    <td>F1</td>
    <td>F1</td>
    <td>F1</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>SOTA</td>
    <td>n/s</td>
    <td></td>
    <td>76.71</td>
    <td>66.40</td>
    <td>70.68</td>
    <td>67.51</td>
    <td>69.53</td>
    <td>74.06</td>
    <td>78.50</td>
    <td>n/s</td>
    <td>73.63</td>
    <td>60.51</td>
    <td>83.68</td>
    <td>77.44</td>
  </tr>
  <tr>
    <td>XLM-RoBERTa-Large</td>
    <td>76.37</td>
    <td>1</td>
    <td>82.26</td>
    <td>76.36</td>
    <td>79.42</td>
    <td>76.35</td>
    <td>76.08</td>
    <td>80.89</td>
    <td>78.31</td>
    <td>75.27</td>
    <td>75.17</td>
    <td>60.03</td>
    <td>88.91</td>
    <td>78.81</td>
  </tr>
  <tr>
    <td>SBERT-Large</td>
    <td>75.43</td>
    <td>2</td>
    <td>78.40</td>
    <td>71.36</td>
    <td>75.14</td>
    <td>72.39</td>
    <td>71.87</td>
    <td>77.72</td>
    <td>78.58</td>
    <td>75.85</td>
    <td>74.20</td>
    <td>60.64</td>
    <td>88.66</td>
    <td>77.41</td>
  </tr>
  <tr>
    <td>MBARTRuSumGazeta</td>
    <td>74.70</td>
    <td>3</td>
    <td>76.06</td>
    <td>68.95</td>
    <td>73.04</td>
    <td>72.34</td>
    <td>71.93</td>
    <td>77.83</td>
    <td>76.71</td>
    <td>73.56</td>
    <td>74.18</td>
    <td>60.54</td>
    <td>87.22</td>
    <td>77.51</td>
  </tr>
  <tr>
    <td>Conversational RuBERT</td>
    <td>74.44</td>
    <td>4</td>
    <td>76.69</td>
    <td>69.09</td>
    <td>73.11</td>
    <td>69.44</td>
    <td>68.68</td>
    <td>75.56</td>
    <td>77.31</td>
    <td>74.40</td>
    <td>73.10</td>
    <td>59.95</td>
    <td>87.86</td>
    <td>77.78</td>
  </tr>
  <tr>
    <td>LaBSE</td>
    <td>74.11</td>
    <td>5</td>
    <td>77.00</td>
    <td>69.19</td>
    <td>73.55</td>
    <td>70.34</td>
    <td>69.83</td>
    <td>76.38</td>
    <td>74.94</td>
    <td>70.84</td>
    <td>73.20</td>
    <td>59.52</td>
    <td>87.89</td>
    <td>78.47</td>
  </tr>
  <tr>
    <td>XLM-RoBERTa-Base</td>
    <td>73.60</td>
    <td>6</td>
    <td>76.35</td>
    <td>69.37</td>
    <td>73.42</td>
    <td>68.45</td>
    <td>67.45</td>
    <td>74.05</td>
    <td>74.26</td>
    <td>70.44</td>
    <td>71.40</td>
    <td>60.19</td>
    <td>87.90</td>
    <td>78.28</td>
  </tr>
  <tr>
    <td>RuBERT</td>
    <td>73.45</td>
    <td>7</td>
    <td>74.03</td>
    <td>66.14</td>
    <td>70.75</td>
    <td>66.46</td>
    <td>66.40</td>
    <td>73.37</td>
    <td>75.49</td>
    <td>71.86</td>
    <td>72.15</td>
    <td>60.55</td>
    <td>86.99</td>
    <td>77.41</td>
  </tr>
  <tr>
    <td>MBART-50-Large-Many-to-Many</td>
    <td>73.15</td>
    <td>8</td>
    <td>75.38</td>
    <td>67.81</td>
    <td>72.26</td>
    <td>67.13</td>
    <td>66.97</td>
    <td>73.85</td>
    <td>74.78</td>
    <td>70.98</td>
    <td>71.98</td>
    <td>59.20</td>
    <td>87.05</td>
    <td>77.24</td>
  </tr>
  <tr>
    <td>SlavicBERT</td>
    <td>71.96</td>
    <td>9</td>
    <td>71.45</td>
    <td>63.03</td>
    <td>68.44</td>
    <td>64.32</td>
    <td>63.99</td>
    <td>71.31</td>
    <td>72.13</td>
    <td>67.57</td>
    <td>72.54</td>
    <td>58.70</td>
    <td>86.43</td>
    <td>77.16</td>
  </tr>
  <tr>
    <td>EnRuDR-BERT</td>
    <td>71.51</td>
    <td>10</td>
    <td>72.56</td>
    <td>64.74</td>
    <td>69.07</td>
    <td>61.44</td>
    <td>60.21</td>
    <td>68.34</td>
    <td>74.19</td>
    <td>69.94</td>
    <td>69.33</td>
    <td>56.55</td>
    <td>87.12</td>
    <td>77.95</td>
  </tr>
  <tr>
    <td>RuDR-BERT</td>
    <td>71.14</td>
    <td>11</td>
    <td>72.79</td>
    <td>64.23</td>
    <td>68.36</td>
    <td>61.86</td>
    <td>60.92</td>
    <td>68.48</td>
    <td>74.65</td>
    <td>70.63</td>
    <td>68.74</td>
    <td>54.45</td>
    <td>87.04</td>
    <td>77.91</td>
  </tr>
  <tr>
    <td>MBART-50-Large</td>
    <td>69.46</td>
    <td>12</td>
    <td>70.91</td>
    <td>62.67</td>
    <td>67.24</td>
    <td>61.12</td>
    <td>60.25</td>
    <td>68.41</td>
    <td>72.88</td>
    <td>68.63</td>
    <td>70.52</td>
    <td>46.39</td>
    <td>86.48</td>
    <td>77.52</td>
  </tr>
</tbody>
</table>

The table shows per-task scores and a macro-average of those scores to determine a models’s position on the leaderboard. For datasets with multiple evaluation metrics (e.g., macro F1 and weighted F1 for RuSentiment), we use an unweighted average of the metrics as the score for the task when computing the overall macro-average. The same strategy for comparing models’ results was applied in the GLUE benchmark.

## Citation
If you find this repository helpful, feel free to cite our publication:

```
@article{Smetanin2021Deep,
  author = {Sergey Smetanin and Mikhail Komarov},
  title = {Deep transfer learning baselines for sentiment analysis in Russian},
  journal = {Information Processing & Management},
  volume = {58},
  number = {3},
  pages = {102484},
  year = {2021},
  issn = {0306-4573},
  doi = {0.1016/j.ipm.2020.102484}
}
```

Dataset:
```
@inproceedings{rogers2018rusentiment,
  title={RuSentiment: An enriched sentiment analysis dataset for social media in Russian},
  author={Rogers, Anna and Romanov, Alexey and Rumshisky, Anna and Volkova, Svitlana and Gronas, Mikhail and Gribov, Alex},
  booktitle={Proceedings of the 27th international conference on computational linguistics},
  pages={755--763},
  year={2018}
}
```