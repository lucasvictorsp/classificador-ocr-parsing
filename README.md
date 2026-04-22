# Classificador de documentos brasileiros

Projeto local em Python para classificar imagens de documentos brasileiros usando
transfer learning com uma CNN leve em PyTorch. O fluxo foi desenhado para
hardware limitado, com treino pequeno, backbone congelado e integracao com
MLflow local.

Classes conhecidas:

- `CNH_Frente`
- `CNH_Verso`
- `RG_Frente`
- `RG_Verso`
- `CPF_Frente`
- `CPF_Verso`

Nao existe uma setima classe treinada chamada `outros`. Durante a inferencia, o
modelo calcula `softmax`, pega a maior probabilidade e rejeita a predicao como
`outros` quando a confianca fica abaixo de um threshold definido no conjunto de
validacao.

## Estrutura

```text
src/
  main.py                         # pipeline antigo de augmentation
  train_classifier.py             # wrapper de treino do classificador
  infer_classifier.py             # wrapper de inferencia do classificador
  batch_infer_classifier.py       # inferencia final em lote pos-treino
  document_classifier/
    batch_inference.py            # avaliacao externa real sem retreino
    data.py                       # discovery, filtros e split sem leakage
    infer.py                      # inferencia com rejeicao
    losses.py                     # loss factory
    metrics.py                    # metricas, threshold e matriz de confusao
    models.py                     # EfficientNet-B0 / MobileNetV3
    train.py                      # pipeline completo de treino
    utils/
      __init__.py                 # reexports de helpers compartilhados
      runtime.py                  # seed, device, JSON e filesystem
      inference.py                # checkpoint e predicao reutilizavel
dataset_augmented/
  CNH_Frente/
  CNH_Verso/
  RG_Frente/
  RG_Verso/
  CPF_Frente/
  CPF_Verso/
  logs/
    mlflow_manifest.csv
Dockerfile
docker-compose.yml
requirements.txt
pyproject.toml
```

O projeto nao foi transformado em biblioteca PyPI. Os scripts podem ser
executados diretamente a partir do workspace.

## Dataset usado no treino

O treino espera a pasta `dataset_augmented/` com uma subpasta por classe.
Somente arquivos `.jpg` sao usados como entrada.

Arquivos ignorados automaticamente:

- nomes terminando em `_mask.jpg`
- arquivos `.txt`
- arquivos `.json`
- qualquer arquivo que nao seja `.jpg`

Exemplo aceito:

```text
dataset_augmented/CNH_Frente/00014699__orig.jpg
dataset_augmented/CNH_Frente/00014699__aug01.jpg
dataset_augmented/CNH_Frente/00014699__orig_mask.jpg   # ignorado
dataset_augmented/CNH_Frente/00014699__orig.txt        # ignorado
dataset_augmented/CNH_Frente/00014699__orig.json       # ignorado
```

## Split sem leakage

O split e feito por ID de origem, nao por arquivo individual. Isso evita que a
imagem original caia em treino e uma imagem aumentada derivada dela caia em
validacao ou teste.

Padrao extraido:

- `00014699__orig.jpg` -> ID de origem `00014699`
- `00014699__aug01.jpg` -> ID de origem `00014699`
- `00014699__aug02.jpg` -> ID de origem `00014699`

Internamente o grupo de split e `classe:ID`, por exemplo
`CNH_Frente:00014699`. O prefixo da classe evita colisoes acidentais caso duas
classes tenham documentos com o mesmo numero de ID.

Proporcao do split:

- treino: 70%
- validacao: 20%
- teste: 10%

O split e estratificado no nivel dos grupos, com seed fixa. A atribuicao final
fica salva em:

```text
artifacts/document_classifier/splits.csv
```

## Estrategia de modelagem

Padrao do treino:

- arquitetura: `efficientnet_b0`
- pesos: ImageNet via Torchvision
- saida final: 6 logits, um por classe conhecida
- input: `224x224`
- loss: `CrossEntropyLoss`
- otimizador: `AdamW`
- batch pequeno: `8` por padrao
- backbone congelado por padrao
- apenas o classificador e o ultimo bloco de features ficam treinaveis
- early stopping por perda de validacao

Tambem existe suporte a `mobilenet_v3_small` para maquinas ainda mais fracas:

```powershell
python src\train_classifier.py --model-name mobilenet_v3_small
```

Se a maquina estiver totalmente offline e os pesos ImageNet ainda nao estiverem
em cache, use `--no-pretrained`. A opcao existe para viabilidade local, mas o
fluxo recomendado e usar EfficientNet-B0 com pesos pre-treinados.

## Threshold para `outros`

O modelo nunca aprende uma classe `outros`. A rejeicao acontece assim:

1. O modelo produz logits para as 6 classes conhecidas.
2. A inferencia aplica `softmax`.
3. A classe candidata e `argmax(probs)`.
4. Se `max(probs) < threshold`, o resultado final vira `outros`.

O threshold e calculado apos o treino usando a distribuicao das probabilidades
maximas no conjunto de validacao. Por padrao, o projeto usa o percentil 5:

```text
threshold = percentile_5(max_softmax_probabilities_validation)
```

Isso significa que cerca de 5% das imagens de validacao com menor confianca
ficam abaixo do limite e seriam rejeitadas. O valor e salvo no checkpoint e em:

```text
artifacts/document_classifier/threshold.json
artifacts/document_classifier/reports/validation_confidence.csv
artifacts/document_classifier/reports/test_confidence.csv
```

Voce pode ajustar o percentil:

```powershell
python src\train_classifier.py --threshold-percentile 10
```

Percentis maiores rejeitam mais imagens. Percentis menores rejeitam menos, mas
aumentam a chance de aceitar uma imagem desconhecida com confianca indevida.

## Instalacao local

Recomendado com `venv` e `pip`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

O `pyproject.toml` tambem contem as dependencias principais. Se preferir Poetry,
gere um lock novo depois das alteracoes de dependencias:

```powershell
poetry lock
poetry install
```

## Treino

Treino padrao local, com MLflow em pasta local:

```powershell
python src\train_classifier.py `
  --dataset-dir dataset_augmented `
  --batch-size 8 `
  --epochs 15 `
  --num-workers 0 `
  --learning-rate 0.001
```

Para CPU ou GPU fraca, mantenha `--batch-size 8` ou reduza para `4`.
Em Windows, `--num-workers 0` costuma ser o modo mais estavel e economico.

Principais artefatos gerados:

```text
artifacts/document_classifier/
  best_model.pt
  label_map.json
  classes.csv
  splits.csv
  threshold.json
  reports/
    classification_report_test.csv
    classification_report_test.txt
    confusion_matrix_test.csv
    confusion_matrix_test.png
    metrics_test.json
    validation_confidence.csv
    test_confidence.csv
```

Metricas minimas reportadas:

- accuracy
- macro F1
- F1 por classe no classification report
- matriz de confusao
- taxa de rejeicao pelo threshold
- accuracy somente nas amostras aceitas pelo threshold

## MLflow

O treino registra:

- parametros: learning rate, batch size, epochs, modelo, input size, seed,
  camadas congeladas, parametros treinaveis/congelados
- metricas por epoca: loss e accuracy de treino, loss, accuracy e macro F1 de
  validacao
- metricas finais de validacao e teste
- checkpoint e artefatos de relatorio
- modelo via `mlflow.pytorch.log_model`
- `dataset_augmented/logs/mlflow_manifest.csv` como artefato, quando existir

Para usar um tracking server local:

```powershell
mlflow server `
  --backend-store-uri sqlite:///mlruns/mlflow.db `
  --default-artifact-root ./mlartifacts `
  --host 127.0.0.1 `
  --port 5000
```

Em outro terminal:

```powershell
python src\train_classifier.py `
  --mlflow-tracking-uri http://127.0.0.1:5000
```

A UI fica em:

```text
http://127.0.0.1:5000
```

## Docker local

Subir MLflow local:

```powershell
docker compose up mlflow
```

Treinar dentro do Docker usando o MLflow do compose:

```powershell
docker compose --profile train up trainer
```

Essa stack e propositalmente simples: Docker + MLflow tracking server local.
Nao inclui Airflow, Prefect ou Grafana para nao aumentar o custo operacional em
maquina fraca. O monitoramento pratico fica nos logs do processo, nos artefatos
salvos e na UI do MLflow.

### Inferencia real pos-treino via Docker

Existe um dataset externo para avaliacao final de inferencia em:

```text
/mnt/d/Lucas/sample_dataset_inference
```

Estrutura esperada:

```text
sample_dataset_inference/
  CNH_Frente/
  CNH_Verso/
  RG_Frente/
  RG_Verso/
  CPF_Frente/
  CPF_Verso/
  Outro/
```

Esse dataset nao e usado em treino, validacao ou teste interno. Ele entra apenas
depois que o modelo ja foi treinado e o checkpoint
`artifacts/document_classifier/best_model.pt` ja existe.

Suba o MLflow:

```powershell
docker compose up -d --build mlflow
```

Depois rode a inferencia real em lote:

```powershell
docker compose --profile inference run --rm real-inference
```

Por padrao, o servico `real-inference` monta o dataset externo como somente
leitura:

```text
/mnt/d/Lucas/sample_dataset_inference -> /app/real_inference_dataset
```

O pipeline percorre todas as subpastas existentes. Pastas vazias nao quebram a
execucao; elas sao registradas como aviso em log e em:

```text
artifacts/document_classifier/real_inference/real_inference_warnings.txt
```

Resultados gerados:

```text
artifacts/document_classifier/real_inference/
  real_inference_predictions.csv
  real_inference_metrics.json
  real_inference_warnings.txt
  real_inference_classification_report.csv
  real_inference_classification_report.txt
  real_inference_confusion_matrix.csv
  real_inference_confusion_matrix.png
```

Cada linha de `real_inference_predictions.csv` contem:

- caminho da imagem
- pasta de origem
- classe real inferida pela pasta
- classe candidata do modelo antes do threshold
- classe final apos threshold
- probabilidade maxima
- threshold usado
- indicador `rejected_by_threshold`
- uma coluna `prob_<classe>` para cada classe conhecida

A pasta `Outro` e tratada como rotulo real `outros` na avaliacao. Isso nao
altera o treinamento: continua nao existindo classe `outros` no modelo.

O pipeline tambem procura arquivos tabulares colocados diretamente na raiz de
`sample_dataset_inference`, como `.csv`, `.tsv`, `.parquet`, `.xlsx` ou `.xls`.
Se encontrar, registra esses arquivos no MLflow mantendo o nome exato.

No MLflow, procure runs com nome parecido com:

```text
real_inference_YYYYMMDD_HHMMSS
```

Metricas registradas quando aplicavel:

- `external_total_images`
- `external_rejection_rate`
- `external_accuracy_final`
- `external_macro_f1_final`
- `external_outros_recall`
- `external_known_accuracy_final`
- `external_empty_or_invalid_folders`
- `external_tabular_files_found`

A UI do MLflow fica em:

```text
http://localhost:5000
```

Para rodar com outro checkpoint ou threshold manual:

```powershell
docker compose --profile inference run --rm real-inference `
  python src/batch_infer_classifier.py `
  --dataset-dir /app/sample_dataset_inference `
  --checkpoint /app/artifacts/document_classifier/best_model.pt `
  --threshold 0.80 `
  --output-dir /app/artifacts/document_classifier/real_inference_threshold_080 `
  --mlflow-tracking-uri http://mlflow:5000
```

Para avaliar outro dataset externo monitorado pelo MLflow, informe o caminho por
variavel de ambiente. Exemplo com:

```text
/mnt/d/Lucas/dataset_augmented_imagem_google
```

No PowerShell:

```powershell
$env:REAL_INFERENCE_DATASET="/mnt/d/Lucas/dataset_augmented_imagem_google"
$env:REAL_INFERENCE_OUTPUT="real_inference_google_images"
$env:REAL_INFERENCE_RUN_NAME="real_inference_google_images"
docker compose --profile inference run --rm real-inference
```

Isso salva os artefatos em:

```text
artifacts/document_classifier/real_inference_google_images/
```

E cria uma run separada no MLflow chamada:

```text
real_inference_google_images
```

Para voltar ao dataset padrao no mesmo terminal:

```powershell
Remove-Item Env:REAL_INFERENCE_DATASET
Remove-Item Env:REAL_INFERENCE_OUTPUT
Remove-Item Env:REAL_INFERENCE_RUN_NAME
```

## Inferencia

Inferencia em uma imagem:

```powershell
python src\infer_classifier.py `
  --input dataset_augmented\CNH_Frente\00014699__orig.jpg `
  --checkpoint artifacts\document_classifier\best_model.pt
```

Inferencia em uma pasta:

```powershell
python src\infer_classifier.py `
  --input dataset_augmented\CNH_Frente `
  --checkpoint artifacts\document_classifier\best_model.pt `
  --output-csv artifacts\document_classifier\predictions.csv `
  --output-json artifacts\document_classifier\predictions.json
```

Exemplo de saida:

```text
dataset_augmented\CNH_Frente\00014699__orig.jpg -> CNH_Frente (pred=CNH_Frente, conf=0.9821, threshold=0.7412)
dataset_augmented\CNH_Frente\foto_ruim.jpg -> outros (pred=RG_Frente, conf=0.4120, threshold=0.7412)
```

Para testar outro threshold sem retreinar:

```powershell
python src\infer_classifier.py `
  --input caminho\imagem.jpg `
  --threshold 0.80
```

## Observacoes operacionais

- Nao use mascaras, `.txt` ou `.json` como input do classificador. O codigo ja
  filtra esses arquivos.
- O manifesto `mlflow_manifest.csv` e registrado no MLflow para rastreabilidade,
  mas o split do classificador e derivado dos arquivos encontrados e dos IDs de
  origem.
- O checkpoint contem arquitetura, tamanho de input, classes e threshold.
- O modelo final tem exatamente 6 saidas. `outros` e apenas uma decisao de
  inferencia quando a confianca e baixa.
- Para melhorar robustez contra desconhecidos reais, avalie imagens externas ao
  dominio e ajuste o percentil/threshold observando o trade-off entre rejeicao e
  erro.
