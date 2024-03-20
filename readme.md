# Projeto de Sinais e Sistemas

Este é o projeto da disciplina de sinais e sistemas que aborda o processamento de sinais de áudio, com foco em análise, transformações e extração de características.

## Sobre

Para este trabalho utilizou-se o dataset FCSS2, mais especificamente seu subset de audios relacionados a natureza que estão divididos em 7 classes.

## Dependências

Para rodar este projeto, é necessaário python3 3.x ealgumas depenências que podem. Para instala-las automaticamente utilize o comando:

```console
pip3 install -r requirements.txt
```

Certifique-se de ter todas as dependências instaladas antes de executar o projeto.

## Funcionalidades

- Visualizar áudio e transformada:
    - Esse script permite que você escolha uma classe para visualizar um exemplo da sua forma de áudio e sua transformada.
    ```console    
    python3 visualize.py Fire # visualizar dados para fogo
    python3 visualize.py Rain # Visualizar dados para chuva
    ```
- preprocessar audios:
    - Esse script cria frames de 1s a partir dos audios de 5 segudos.
    - os audios são salvos na pasta 'frames' e um novo csv 'frames.csv' é criado
    ```console    
    python3 preprocess # preprocessa os dados
    ```
- Executar MFCC ou FFT
    - Atualmente, o modelo utiliza o MFCC. Para utilizar somente a FFT, mudar a variável 'use_mfcc' em 'Feature_extraction.py' para False
    
- Executar modelo
    - No momento esse script executa uma Random Forest
    ```console    
    python3 main.py # classificar dados
    ```




