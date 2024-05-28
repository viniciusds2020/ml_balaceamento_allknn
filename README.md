# Balanceamento de Dados com AllKNN

Este repositório contém um código de Machine Learning que utiliza o algoritmo AllKNN do pacote `imblearn` para realizar o balanceamento de dados. O balanceamento de dados é uma etapa crucial em muitos projetos de Machine Learning, especialmente quando se lida com conjuntos de dados desbalanceados, onde uma ou mais classes estão sub-representadas.

## Descrição

O algoritmo AllKNN é uma técnica de undersampling que remove amostras da classe majoritária com base no algoritmo k-vizinhos mais próximos (k-NN), ajudando a criar um conjunto de dados mais equilibrado.

## Requisitos

- Python 3.7+
- scikit-learn
- imbalanced-learn (imblearn)

Você pode instalar os requisitos usando o pip:

```bash
pip install -r requirements.txt
```

## Estrutura do Projeto

```
.
├── data
│   ├── input.csv        # Dados de entrada
├── notebooks
│   └── data_balancing.ipynb # Jupyter Notebook com o código
├── README.md
└── requirements.txt
```

## Como Usar

### 1. Preparar os Dados

Coloque seus dados de entrada no arquivo `data/input.csv`. Certifique-se de que o arquivo esteja formatado corretamente com a coluna de rótulos devidamente identificada.

### 2. Executar o Balanceamento

Você pode executar o balanceamento diretamente através do script Python:

```bash
python src/balance_data.py
```

### 3. Visualizar os Resultados

Os dados balanceados serão salvos em `data/balanced_data.csv`. Você pode carregar este arquivo para verificar os resultados.

### 4. Jupyter Notebook

Alternativamente, você pode explorar e executar o código no Jupyter Notebook disponível em `notebooks/data_balancing.ipynb`.

## Exemplo de Código

Aqui está um exemplo simplificado de como utilizar o AllKNN no Python:

```python
import pandas as pd
from imblearn.under_sampling import AllKNN

# Carregar os dados
data = pd.read_csv('data/input.csv')
X = data.drop('label', axis=1)
y = data['label']

# Aplicar o AllKNN
allknn = AllKNN()
X_resampled, y_resampled = allknn.fit_resample(X, y)

# Salvar os dados balanceados
balanced_data = pd.concat([X_resampled, y_resampled], axis=1)
balanced_data.to_csv('data/balanced_data.csv', index=False)
```

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Acknowledgements

Este projeto utiliza a biblioteca [imbalanced-learn](https://imbalanced-learn.org/stable/) para técnicas de balanceamento de dados.
