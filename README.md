# Biblioteca MLOps com MLflow

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-em%20desenvolvimento-orange)

Uma biblioteca Python modular e extensível projetada para simplificar e padronizar o ciclo de vida de Machine Learning (MLOps) utilizando o **MLflow**. Ela fornece um framework robusto para rastreamento de experimentos, gerenciamento de modelos e integração com pipelines de ML.

## ✨ Funcionalidades Principais

-   🚀 **Pipeline de Orquestração:** Executa fluxos de treinamento, avaliação e registro de modelos de ponta a ponta.
-   🛠️ **Modularidade:** Separação clara de responsabilidades entre `Trainers` e `Evaluators`, permitindo fácil extensão para novos modelos e métricas.
-   📊 **Rastreamento com MLflow:** Encapsula as chamadas ao MLflow para log de parâmetros, métricas e artefatos de forma organizada.
-   📦 **Gerenciamento de Modelos:** Facilita o registro, versionamento e promoção de modelos no MLflow Model Registry.
-   📝 **Logging Aprimorado:** Logs claros e informativos com emojis e separadores para facilitar o acompanhamento do processo.
-   ⚙️ **Configuração Centralizada:** Carrega parâmetros de um arquivo `config.yaml`, mantendo o código limpo e parametrizável.

## 📂 Estrutura do Projeto

```
mlops-project/
├── mlops_lib/          # Código fonte da biblioteca
├── trainers/           # Classes para treinamento de modelos
├── evaluators/         # Classes para avaliação de modelos
├── utils/              # Módulos de utilidades
├── config/
│   └── config.yaml
├── main.py             # Script principal de exemplo e teste local
├── README.md
├── requirements.txt
└── setup.py            # Script para empacotamento
```

## 🚀 Instalação

### 1. A partir de um Pacote Wheel (Recomendado para Databricks)

Gere o arquivo `.whl` a partir da raiz do projeto:
```bash
python setup.py bdist_wheel
```
Em seguida, instale o arquivo gerado que estará na pasta `dist/`:
```bash
pip install dist/mlops_lib_demo-0.1.0-py3-none-any.whl
```

### 2. Para Desenvolvimento (Modo Editável)

Para instalar a biblioteca de forma que as alterações no código fonte sejam refletidas imediatamente, use o modo editável:
```bash
pip install -e .
```

## ⚡ Guia Rápido de Uso

Abaixo um exemplo de como usar a biblioteca para treinar e registrar um modelo.

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Importando componentes da biblioteca
from mlops_lib.pipeline_integration import PipelineIntegrator
from mlops_lib.experiment_tracking import ExperimentTracker
from mlops_lib.model_management import ModelManager
from trainers.classification_trainer import ClassificationTrainer
from evaluators.classification_evaluator import ClassificationEvaluator

# 1. Configurar os componentes principais
tracker = ExperimentTracker(
    tracking_uri="mlruns",
    experiment_name="MyAwesomeExperiment"
)
manager = ModelManager(tracking_uri="mlruns")
pipeline_integrator = PipelineIntegrator(tracker, manager)

# 2. Preparar os dados
X, y = make_classification()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Executar a pipeline
pipeline_integrator.run_training_pipeline(
    trainer=ClassificationTrainer(),
    evaluator=ClassificationEvaluator(),
    X_train=pd.DataFrame(X_train), y_train=pd.Series(y_train),
    X_test=pd.DataFrame(X_test), y_test=pd.Series(y_test),
    model_name="MyClassifier",
    params={"C": 1.0, "solver": "liblinear"},
    model_flavor="sklearn",
    run_name="first_classifier_run",
    register_threshold=0.9,
    threshold_metric="roc_auc"
)
```

## 🧩 Estendendo a Biblioteca

Para adicionar suporte a um novo tipo de modelo (ex: `LightGBM`), basta:
1.  Criar um `LightGBMTrainer` em `trainers/`.
2.  Criar um `LightGBMEvaluator` em `evaluators/`.
3.  Importar e usá-los na sua pipeline.

A estrutura foi projetada para ser facilmente extensível sem a necessidade de modificar o código principal da biblioteca.


## 📜 Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
