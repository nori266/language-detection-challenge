## Evaluation

Inside this folder you can find testsets for tasks 1 and 2, respectively:

- langid.test
- langid-variants.test

You need to generate a file with the corresponding labels for these tasks. This is, if your 
test file has the following senteces in taks 1:

```
This is an English sentence
Esta es una frase en español
Isto é uma frase em português
```

You should generate a file with your hypothesis:
```
en
es
pt
```

In the same way, for task2, given the input:

```
Tem gente no Barcelona que vê Philippe Coutinho como um possível substituto de Neymar. 
Belga estará perto de renovar, mas, enquanto tal não sucede, o Chelsea teme que este mude de ideias e rume ao Real Madrid.
```

Your output file should contain the hypothesis:
```
pt-br
pt-pt
```



## Training data

Access all langid data inside https://drive.google.com/drive/folders/1a4Gzu5-vsMAjMUnObG5lm5Ym591Z88JL?usp=sharing

Inside the data folder you will find data for the languages we are focusing on. Each is a monolingual file of a particular language:

* EN - data.en
* ES - data.es
* PT-PT - data.pt-pt
* PT-BR - data.pt-br

Each dataset has also an aditional .ids file which has the origin of the data. These datasets should suffice for the tasks 1 and 2, but
you are welcome to use additional data if you want to. The provided data is in plain UTF-8 text format.

**Tip: The data files provided are very large (for EN,ES and PT), so you might want to subsample a smaller set to train your system**

## Suggested reading

http://aclweb.org/anthology/P15-2063

