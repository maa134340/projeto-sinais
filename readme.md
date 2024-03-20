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

- Escolher entre classes
    - Há dois arquivos de metadados, um somente com a classe de sons ambiente ('metadata_environmental_class.csv') e outro também com a classe de ameaças para a floresta('metadata_environmental_and_threats_class.csv')
    - Por padrão, apenas o com a classe de sons ambiente está escolhida, por causa do tempo de execução do modelo, mas é possível mudar isso no arquivo main.py
    ```main.py    
    preprocess = Preprocess(metadata_file="metadata_environmental_and_threats_class.csv")
    ```

- Visualizar áudio e transformada:
    - Esse script permite que você escolha uma classe para visualizar um exemplo da sua forma de áudio e sua transformada.
    ```console    
    python3 visualize.py Fire # visualizar dados para fogo
    python3 visualize.py Rain # Visualizar dados para chuva
    ```
- Executar MFCC ou FFT
    - Atualmente, o modelo utiliza o MFCC. Para utilizar somente a FFT, mudar a variável 'use_mfcc' em 'Feature_extraction.py' para False

- Executar Projeto
    - Esse script carrega, processa e classifica os áudios
    ```console    
    python3 main.py 
    ```




