## Python implementation of: "Performance Prediction via Bayesian Matrix Factorisation for Multilingual Natural Language Processing Tasks"

This repository contains a Bayesian approach to performance prediction for multilingual NLP tasks. This is the implementation for [Performance Prediction via Bayesian Matrix Factorisation for Multilingual Natural Language Processing Tasks](https://aclanthology.org/2023.eacl-main.131/)

Each src-folder contains the code for either MF, PMF, BPMF or BPMF CTX implementation.
In every scenario, the folder contains a main file to be run and parameters can be set via the utils-file.

To be comparable with [NLPerf](https://github.com/xiamengzhou/NLPerf), we use and compare both approaches using the same dataset splits given in folder "data/NLPerfSplit/". 
As the tests and comparisons can be performed for a maximum of 10 runs, 10 splits are available.
Complete files of performance scores and features can be found in the folder "data".

Visualization of the main idea of MF

![text](images/MF.PNG) 

### Citation  
If you find this repository useful, please consider citing our work:
```
@inproceedings{schram2023performance,
  title={Performance Prediction via Bayesian Matrix Factorisation for Multilingual Natural Language Processing Tasks},
  author={Schram, Viktoria and Beck, Daniel and Cohn, Trevor},
  booktitle={Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics},
  pages={1790--1801},
  year={2023}
}
```
This work is inspired by [NLPerf](https://github.com/xiamengzhou/NLPerf) and [NLP-Bilingual-Task-Performance-Prediction](https://github.com/tianzhipengfei/NLP-Bilingual-Task-Performance-Prediction).  


<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

