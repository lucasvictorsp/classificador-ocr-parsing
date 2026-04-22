# Document Classifier

Este diretório contém o subprojeto de classificação de imagens de documentos brasileiros.

## Visão geral do projeto

O classificador recebe imagens de documentos e prediz uma das classes conhecidas:

```text
CNH_Frente
CNH_Verso
RG_Frente
RG_Verso
CPF_Frente
CPF_Verso
```

Ele foi construído como uma etapa intermediária do sistema: depois que o dataset é preparado/aumentado por `src/data_augmentation/`, o classificador aprende a distinguir tipo e lado do documento. A saída do classificador pode então orientar a etapa `src/ocr_parsing/`, que pressupõe que o tipo do documento já é conhecido.

O modelo não treina uma classe `outros`. A rejeição de documentos desconhecidos é feita em inferência: se a maior probabilidade softmax ficar abaixo de um threshold calculado na validação, o resultado final é convertido para `outros`.

Entradas principais:

- dataset organizado por pasta de classe;
- imagens `.jpg`;
- checkpoint `best_model.pt` para inferência;
- dataset (preferencialmente externo para ver se o modelo consegue ser generalista) opcional para avaliação final em lote.

Saídas principais:

- checkpoint treinado;
- mapeamento de classes;
- split auditável;
- métricas de treino/validação/teste;
- matriz de confusão;
- análise de confiança e rejeição;
- runs e artefatos no MLflow.

## Arquitetura geral

```text
dataset_augmented/
  <classe>/
    <id>__orig.jpg
    <id>__aug01.jpg
    <id>__orig_mask.jpg   -> ignorado
    <id>__orig.txt        -> ignorado
    <id>__orig.json       -> ignorado
        |
        v
+--------------------------+
| data.discover_samples    |
| filtra imagens .jpg e    |
| remove máscaras          |
+--------------------------+
        |
        v
+--------------------------+
| data.split_samples       |
| split por group_id       |
| classe:documento_origem  |
+--------------------------+
        |
        v
+--------------------------+
| DocumentImageDataset     |
| PIL -> RGB -> transforms |
+--------------------------+
        |
        v
+--------------------------+
| models.build_model       |
| EfficientNet-B0 ou       |
| MobileNetV3 Small        |
+--------------------------+
        |
        v
+--------------------------+
| train.py                 |
| treino, validação,       |
| early stopping, teste    |
+--------------------------+
        |
        v
+--------------------------+
| metrics.py + MLflow      |
| reports, threshold,      |
| matriz, artefatos        |
+--------------------------+
        |
        v
artifacts/document_classifier/
```

Para inferência:

```text
imagem ou diretório
        |
        v
infer.py / batch_inference.py
        |
        v
checkpoint + metadata
        |
        v
probabilidades por classe
        |
        v
threshold de rejeição
        |
        v
classe final ou "outros"
```

## Fluxo de execução de treinamento

1. `src/train_classifier.py` chama `document_classifier.train.main()`.
2. `parse_args()` lê hiperparâmetros, caminhos, MLflow, modelo, device e opções de transfer learning.
3. `set_seed()` fixa seeds de Python, NumPy e PyTorch.
4. `select_device()` escolhe `cuda` se disponível e `--device auto` estiver em uso; caso contrário usa o device solicitado.
5. `discover_samples()` percorre `dataset_dir` e coleta imagens válidas.
6. `split_samples()` cria split 70/20/10 por grupo de origem, evitando vazamento de augmentations entre treino, validação e teste.
7. `write_split_csv()` salva o split em `artifacts/document_classifier/splits.csv`.
8. `build_transforms()` cria transformações de treino e avaliação.
9. `DocumentImageDataset` carrega imagens com PIL, converte para RGB e aplica transforms.
10. `build_model()` cria uma arquitetura Torchvision com pesos ImageNet por padrão.
11. `build_loss()` cria a loss configurada, atualmente `CrossEntropyLoss`.
12. O otimizador `AdamW` treina apenas parâmetros com `requires_grad=True`.
13. A cada época, `train_one_epoch()` treina e `evaluate_model()` valida.
14. `ReduceLROnPlateau` reduz learning rate quando a perda de validação para de melhorar.
15. O melhor checkpoint por `val_loss` é salvo como `best_model.pt`.
16. Early stopping interrompe treino após `patience` épocas sem melhora.
17. O melhor checkpoint é recarregado.
18. O modelo é avaliado em validação e teste.
19. `choose_rejection_threshold()` define o threshold pelo percentil das confianças máximas de validação.
20. Relatórios, matrizes, JSONs, CSVs e artefatos MLflow são salvos.

## Fluxo de inferência

Há duas formas de inferência:

- `infer.py`: inferência simples em uma imagem ou diretório de `.jpg`;
- `batch_inference.py`: inferência final em dataset externo, com métricas e MLflow.

Em ambos os casos:

1. O checkpoint é carregado com `torch.load()`.
2. A arquitetura é reconstruída por `build_model()` usando metadados do checkpoint.
3. O `state_dict` é carregado.
4. A imagem é convertida para RGB, redimensionada, transformada em tensor e normalizada.
5. O modelo retorna logits.
6. `torch.softmax()` produz probabilidades.
7. A classe candidata é o índice de maior probabilidade.
8. Se a confiança máxima for menor que o threshold, a classe final vira `outros`.

## Estrutura de pastas e arquivos

```text
src/document_classifier/
  __init__.py
  batch_inference.py
  constants.py
  data.py
  infer.py
  losses.py
  metrics.py
  models.py
  readme.md
  train.py
  utils/
    __init__.py
    inference.py
    runtime.py
```

## Explicação dos arquivos Python

### `constants.py`

Contém constantes globais:

- `KNOWN_CLASSES`: classes oficiais do modelo.
- `IMAGENET_MEAN`: média usada na normalização de entrada.
- `IMAGENET_STD`: desvio padrão usado na normalização de entrada.

### `data.py`

Responsável por descoberta de amostras, split e datasets PyTorch.

- `ImageSample`: dataclass imutável que representa uma imagem, seu label, índice numérico, `origin_id` e `group_id`.
- `DocumentImageDataset`: dataset PyTorch que carrega imagens com PIL e retorna `(tensor, label_index)`.
- `extract_origin_id()`: extrai o ID de origem de nomes como `<id>__orig.jpg` e `<id>__aug01.jpg`.
- `is_training_image()`: aceita apenas `.jpg` que não terminem com `_mask.jpg`.
- `discover_samples()`: percorre as pastas de classe e cria `ImageSample`.
- `samples_to_frame()`: converte amostras em `DataFrame` para auditoria e split.
- `_can_stratify()`: verifica se há quantidade suficiente de grupos por classe para estratificação.
- `split_samples()`: divide grupos em treino, validação e teste sem vazamento entre original e augmentations.
- `build_transforms()`: cria pipelines de transforms para treino e avaliação.
- `write_split_csv()`: grava o split em CSV.

### `models.py`

Constrói modelos de transfer learning.

- `ModelInfo`: dataclass com modelo e contagens de parâmetros congelados, treináveis e totais.
- `_count_parameters()`: conta parâmetros totais, treináveis ou congelados.
- `_set_requires_grad()`: liga ou desliga gradiente de todos os parâmetros de um módulo.
- `_replace_classifier()`: troca a camada final para produzir `num_classes` logits.
- `_unfreeze_last_feature_blocks()`: descongela os últimos blocos do extrator de features.
- `build_model()`: cria `efficientnet_b0` ou `mobilenet_v3_small`, aplica pesos ImageNet opcionais e define congelamento.
- `load_state_dict()`: carrega pesos em um modelo.

### `losses.py`

Contém a fábrica de loss.

- `build_loss()`: retorna `nn.CrossEntropyLoss()` quando `loss_name="cross_entropy"`. Outros nomes geram `ValueError`.

### `metrics.py`

Implementa avaliação e relatórios.

- `EvaluationResult`: dataclass com loss, accuracy, macro F1, labels reais, predições, probabilidade máxima e matriz de probabilidades.
- `evaluate_model()`: avalia um split, calcula softmax e agrega arrays NumPy.
- `choose_rejection_threshold()`: escolhe threshold pelo percentil das confianças máximas.
- `classification_report_frame()`: gera `classification_report` do scikit-learn como `DataFrame`.
- `save_confusion_matrix()`: salva matriz de confusão em CSV e PNG.
- `save_confidence_analysis()`: salva análise por amostra com aceitação/rejeição e retorna métricas agregadas.

### `train.py`

Ponto de entrada de treinamento.

- `parse_args()`: define argumentos de dataset, saída, modelo, hiperparâmetros, device e MLflow.
- `make_loader()`: cria `DataLoader` com `pin_memory` quando CUDA está disponível.
- `train_one_epoch()`: executa uma época de treino com forward, loss, backward e step.
- `save_checkpoint()`: grava `model_state_dict` e `metadata`.
- `main()`: executa o ciclo completo de treino, validação, teste, threshold, relatórios e MLflow.

### `infer.py`

Inferência simples.

- `parse_args()`: lê imagem/diretório, checkpoint, threshold, device e saídas opcionais.
- `iter_input_images()`: coleta uma imagem ou `.jpg` válidos recursivamente.
- `main()`: executa inferência e grava JSON/CSV opcionais.

As funções compartilhadas de carregamento de checkpoint e predição por imagem
ficam em `utils/inference.py`. Assim, `infer.py` e `batch_inference.py`
reutilizam a mesma implementação sem acoplamento entre pipelines.

### `batch_inference.py`

Avaliação final em lote sobre dataset externo.

- `parse_args()`: define dataset externo, checkpoint, output, MLflow e threshold.
- `configure_logging()`: configura logs.
- `is_inference_image()`: aceita imagens por extensão e remove máscaras.
- `normalize_true_label()`: mapeia pasta conhecida para classe ou `Outro`/`outros` para `outros`.
- `discover_dataset_images()`: percorre subpastas imediatas e tolera pastas vazias com warnings.
- `discover_tabular_files()`: identifica arquivos tabulares na raiz do dataset externo.
- `build_prediction_rows()`: executa `predict_image()` e achata probabilidades em linhas CSV.
- `save_external_confusion_matrix()`: salva matriz para labels finais, incluindo `outros`.
- `compute_external_metrics()`: calcula métricas agregadas quando há ground truth pela pasta.
- `save_external_reports()`: grava predições, métricas, classification report e matriz.
- `main()`: executa inferência externa e registra artefatos no MLflow.


### `utils/runtime.py`

Utilitários genéricos de execução:

- `ensure_dir()`: cria diretórios.
- `set_seed()`: fixa seeds e configura cuDNN determinístico.
- `select_device()`: escolhe `auto`, `cpu`, `cuda` ou outro device PyTorch.
- `write_json()`: grava JSON UTF-8 indentado.
- `read_json()`: lê JSON.
- `to_project_path()`: serializa caminhos em formato POSIX.

### `utils/inference.py`

Utilitários compartilhados por inferência simples e inferência em lote:

- `load_checkpoint_model()`: reconstrói arquitetura e carrega checkpoint.
- `predict_image()`: retorna label predito, label final, confiança, threshold e probabilidades por classe.

---

## Pré-processamento das Imagens

O modelo de classificação de documentos não utiliza máscaras nem anotações estruturadas como entrada. Ele opera diretamente sobre as imagens, sendo fortemente dependente de um pipeline de pré-processamento bem definido que foi aplicado via Torchvision:


Treino:

```text
Resize(input_size * 1.14)
RandomResizedCrop(input_size, scale=(0.86, 1.0), ratio=(0.90, 1.10))
RandomRotation(degrees=4)
ColorJitter(brightness=0.08, contrast=0.08, saturation=0.04)
ToTensor()
Normalize(IMAGENET_MEAN, IMAGENET_STD)
```

Avaliação e inferência:

```text
Resize((input_size, input_size))
ToTensor()
Normalize(IMAGENET_MEAN, IMAGENET_STD)
```

Impactos:

- `Resize` e `RandomResizedCrop` padronizam entrada e criam leve variação espacial.
- `RandomRotation(4)` adiciona robustez a pequenas inclinações sem distorcer demais documentos.
- `ColorJitter` simula iluminação/captura diferentes em baixa intensidade.
- A normalização ImageNet é compatível com pesos pré-treinados do Torchvision.

Não há binarização, retificação, filtros OpenCV ou OCR neste subprojeto.

### EsplicaÇão Detalhada do pré-processamento

Esse pipeline tem dois objetivos principais:

- Padronizar as imagens (tamanho, escala e distribuição de pixels)  
- Aumentar a robustez do modelo (simulando pequenas variações reais)  

A seguir está a descrição detalhada de cada etapa aplicada durante o treinamento:

---

#### Redimensionamento Inicial

Antes de qualquer transformação, a imagem é levemente ampliada.

##### Objetivo

- Criar uma margem para permitir cortes aleatórios posteriormente sem perda excessiva de conteúdo  

##### Intuição

É como “dar um zoom leve” na imagem para que, ao recortar, ainda exista informação suficiente.

---

#### Recorte Aleatório com Redimensionamento

Uma região aleatória da imagem é selecionada e redimensionada para o tamanho padrão de entrada do modelo.

##### Objetivo

- Forçar o modelo a aprender características relevantes independentemente da posição exata do documento na imagem  

**O que isso resolve na prática:**

- Documentos não centralizados  
- Pequenos cortes nas bordas  
- Variações de enquadramento  

---

#### Pequena Rotação Aleatória

A imagem sofre uma leve rotação (alguns graus para esquerda ou direita).

##### Objetivo

- Simular fotos tiradas com o celular em ângulo levemente inclinado  

##### Importância

- Evita que o modelo dependa de documentos perfeitamente alinhados  

---

#### Variação de Cor (Brilho, Contraste e Saturação)

Pequenas alterações são feitas na aparência da imagem:

- Brilho (mais claro ou mais escuro)  
- Contraste (diferença entre tons)  
- Saturação (intensidade das cores)  

##### Objetivo

- Simular diferentes condições de iluminação e qualidade de câmera  

##### Exemplos reais:

- Foto tirada à noite  
- Ambiente com sombra  
- Câmera de baixa qualidade  

---

#### Conversão para Tensor

A imagem é convertida para um formato numérico que pode ser processado pelo modelo.

##### Objetivo

- Transformar os pixels da imagem em uma estrutura matemática (matriz) que o modelo consegue interpretar  

---

#### Normalização

Os valores dos pixels são ajustados para seguir um padrão estatístico (baseado no ImageNet).

##### Objetivo

- Estabilizar o treinamento  
- Acelerar a convergência  
- Tornar o modelo mais eficiente  

##### Intuição

É como “padronizar a escala” dos dados para evitar que valores muito grandes ou muito pequenos atrapalhem o aprendizado.

---

### Por que esse pipeline é importante?

Esse conjunto de transformações foi cuidadosamente escolhido para:

- Melhorar a generalização do modelo  
- Reduzir overfitting  
- Tornar o classificador robusto a variações do mundo real  

Mesmo com um dataset originalmente “quase perfeito”, esse pipeline simula imperfeições comuns encontradas em produção, como:

- Inclinação do documento  
- Variação de iluminação  
- Pequenos cortes  
- Mudanças de enquadramento  

Note que é possível desligar parte desse processo de transformações. Isso se porque se a etapa de data augmentation tiver sido bem aplicada, muitas dessas transformações podem ser desligadas, assim como é feito na etapa de inferência.

---

### Observação Importante

Essas transformações são aplicadas apenas durante o treinamento. Durante a inferência (uso real do modelo), são aplicadas apenas etapas determinísticas (como redimensionamento e normalização), garantindo previsibilidade nas respostas.

Note que é possível desligar parte desse processo de transformações também para o treinamento. Isso se porque se a etapa de data augmentation tiver sido bem aplicada, muitas dessas transformações podem ser desligadas, assim como é feito na etapa de inferência.

## Transfer learning e treinamento

Modelo padrão:

```text
efficientnet_b0
```

Modelo alternativo suportado:

```text
mobilenet_v3_small
```

Por padrão, `build_model()` usa pesos ImageNet (`pretrained=True`) e congela o backbone (`freeze_backbone=True`). O fluxo é:

1. carregar arquitetura Torchvision com pesos ImageNet;
2. congelar todos os parâmetros se `freeze_backbone=True`;
3. substituir a camada classificadora final para `num_classes`;
4. tornar o classificador treinável;
5. descongelar os últimos `train_last_blocks` blocos de `model.features` quando o backbone foi congelado.

Configuração padrão relevante:

```text
model_name = efficientnet_b0
pretrained = True
freeze_backbone = True
train_last_blocks = 1
num_classes = 6
```

Camadas congeladas:

- praticamente todo o extrator de features, quando `freeze_backbone=True`;
- exceto os últimos blocos definidos por `train_last_blocks`.

Camadas treinadas:

- classificador final recém-substituído;
- últimos blocos de features liberados por `_unfreeze_last_feature_blocks()`.

Ao usar `--no-freeze-backbone`, o código não congela o modelo antes de substituir o classificador; portanto, todos os parâmetros permanecem treináveis. Ao usar `--no-pretrained`, os pesos ImageNet não são carregados.

Justificativas e implicações:

- Transfer learning reduz custo computacional e necessidade de muitos dados.
- Congelar o backbone diminui risco de overfitting em datasets pequenos.
- Treinar apenas o classificador pode causar underfitting se o domínio visual for muito diferente do ImageNet.
- Descongelar os últimos blocos é um meio-termo: permite adaptação ao domínio de documentos com menor custo que fine-tuning completo.
- EfficientNet-B0 é uma arquitetura leve para boa relação custo/desempenho.
- MobileNetV3 Small é opção ainda mais econômica, mas pode ter menor capacidade.

## Métricas, avaliação e artefatos

Artefatos principais em `artifacts/document_classifier/`:

```text
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

Métricas por época logadas no MLflow:

- `train_loss`;
- `train_accuracy`;
- `val_loss`;
- `val_accuracy`;
- `val_macro_f1`.

Métricas finais:

- `best_epoch`;
- `best_val_loss`;
- `val_loss`;
- `val_accuracy`;
- `val_macro_f1`;
- `test_loss`;
- `test_accuracy`;
- `test_macro_f1`;
- `threshold`;
- `validation_rejection_rate`;
- `test_rejection_rate`;
- `test_accepted_accuracy`.

O threshold é calculado pelo percentil configurado de `val_result.max_probs`. O padrão é percentil 5, ou seja, o limite fica na região inferior das confianças de validação.

## MLflow

O treinamento configura:

```text
tracking_uri padrão: file:./mlruns
experiment padrão: brazilian_document_classifier
```

Parâmetros logados incluem arquitetura, input size, batch size, learning rate, weight decay, seed, loss, percentil de threshold, uso de pesos pré-treinados, congelamento, blocos treináveis e contagens de parâmetros.

Artefatos logados:

- manifesto `dataset_augmented/logs/mlflow_manifest.csv`, se existir;
- todo `output_dir` em `training_artifacts`;
- modelo PyTorch em `model`.

Na inferência externa em lote, o MLflow recebe:

- parâmetros do modo `real_inference_batch`;
- métricas externas;
- artefatos da pasta de saída;
- arquivos tabulares encontrados na raiz do dataset externo.

## Execução local

Instalação:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Treinamento padrão:

```powershell
python src\train_classifier.py `
  --dataset-dir dataset_augmented `
  --output-dir artifacts\document_classifier `
  --batch-size 8 `
  --epochs 15 `
  --learning-rate 0.001 `
  --num-workers 0
```

Treinamento com MobileNet:

```powershell
python src\train_classifier.py `
  --dataset-dir dataset_augmented `
  --model-name mobilenet_v3_small
```

Treinamento sem pesos ImageNet:

```powershell
python src\train_classifier.py `
  --dataset-dir dataset_augmented `
  --no-pretrained
```

Fine-tuning completo:

```powershell
python src\train_classifier.py `
  --dataset-dir dataset_augmented `
  --no-freeze-backbone
```

Inferência em uma imagem:

```powershell
python src\infer_classifier.py `
  --input dataset_augmented\CNH_Frente\00014699__orig.jpg `
  --checkpoint artifacts\document_classifier\best_model.pt
```

Inferência em diretório com saídas:

```powershell
python src\infer_classifier.py `
  --input dataset_augmented\CNH_Frente `
  --checkpoint artifacts\document_classifier\best_model.pt `
  --output-json artifacts\document_classifier\predictions.json `
  --output-csv artifacts\document_classifier\predictions.csv
```

Inferência externa em lote:

```powershell
python src\batch_infer_classifier.py `
  --dataset-dir caminho\dataset_inferencia `
  --checkpoint artifacts\document_classifier\best_model.pt `
  --output-dir artifacts\document_classifier\real_inference
```

## Docker, compose e YAML

Na raiz há `Dockerfile` e `docker-compose.yml`.

Arquivos de configuração relacionados:

- `requirements.txt`: contém as dependências instaladas com `pip` e usadas pela imagem Docker. Para o classificador, os pacotes centrais são `torch`, `torchvision`, `scikit-learn`, `pandas`, `numpy`, `Pillow`, `matplotlib`, `mlflow` e `tqdm`.
- `pyproject.toml`: define metadados do workspace, faixa de Python (`>=3.10,<3.14`), dependências principais, dependências opcionais de desenvolvimento e regras do Ruff. O `package-mode = false` deixa explícito que o projeto não está configurado como pacote Poetry distribuível.
- `poetry.lock`: fixa versões resolvidas quando o ambiente é criado via Poetry. Não é lido diretamente pelo código de treino, mas auxilia reprodutibilidade.
- `.dockerignore`: remove ambiente virtual, caches Python, artefatos, MLflow local, dataset aumentado e `.git/` do contexto Docker.
- `Dockerfile`: define a imagem Python usada pelos serviços do compose.
- `docker-compose.yml`: YAML com serviços de MLflow, treino e inferência externa.

O `Dockerfile` cria uma imagem Python 3.12 com dependências de OpenCV, Tesseract e bibliotecas Python. Embora Tesseract não seja usado pelo classificador, a imagem é compartilhada com OCR/parsing.

Serviços relevantes do `docker-compose.yml`:

- `mlflow`: sobe MLflow em `http://localhost:5000`, com backend SQLite em `/app/mlruns/mlflow.db` e artefatos em `/app/mlartifacts`.
- `trainer`: perfil `train`, depende de `mlflow`, monta o workspace e monta `/mnt/d/Lucas/dataset_augmented` em `/app/dataset_augmented`.
- `real-inference`: perfil `inference`, monta dataset externo em `/app/real_inference_dataset` como somente leitura.

Subir MLflow:

```powershell
docker compose up -d --build mlflow
```

Treinar no Docker:

```powershell
docker compose --profile train up trainer
```

Rodar inferência externa:

```powershell
docker compose --profile inference run --rm real-inference
```

O projeto foi desenvolvido em Windows e testado com Docker para reduzir problemas de ambiente e deploy. Os caminhos `/mnt/d/...` indicam uso provável de montagem Windows/WSL no Docker. Se esses caminhos não existirem na máquina atual, os volumes devem ser ajustados antes da execução.

Observação sobre o compose: o serviço `trainer` chama `python src/train_classifier.py`, que existe no workspace e é um wrapper fino para `document_classifier.train.main()`.

## Ambiente e portabilidade

O código é local e não está organizado como pacote distribuível. A execução depende do diretório do projeto e dos imports a partir de `src/`. Os wrappers em `src/` simplificam esse uso.

Boas escolhas para Windows:

- `--num-workers 0` evita problemas comuns de multiprocessing no `DataLoader`;
- `Path` é usado em argumentos e artefatos;
- Docker fornece ambiente Linux reprodutível;
- MLflow pode rodar localmente em arquivo ou como serviço no compose.

## Limitações do processo

- Só imagens `.jpg` entram no treinamento; `.png` e outros formatos não são considerados por `is_training_image()`.
- O modelo não aprende a classe `outros`; rejeição depende apenas de threshold de confiança.
- Softmax pode ser excessivamente confiante em exemplos fora de distribuição.
- O split é feito por grupo de origem, mas pressupõe que nomes de arquivos seguem a convenção com `__orig` e `__augNN`.
- A estratificação é usada apenas quando há exemplos suficientes por classe.
- Não há validação automática de balanceamento após augmentação além do split salvo.
- O checkpoint depende dos metadados gravados corretamente.
- `torch.load()` carrega checkpoints locais; não há assinatura/verificação criptográfica.
- A avaliação externa infere ground truth pelo nome da pasta, o que pode mascarar erros se a estrutura estiver incorreta.

## Melhorias futuras

- Adicionar calibração explícita de probabilidades, como temperature scaling.
- Treinar uma estratégia dedicada para desconhecidos com dados `outros`, se houver dados reais suficientes.
- Adicionar suporte a `.png`, `.jpeg` e outros formatos no treinamento, se necessário.
- Registrar curvas de learning rate, precision/recall por classe e exemplos de erro.
- Adicionar validação de distribuição por classe antes do treino.
- Permitir configuração YAML para hiperparâmetros.
- Criar testes unitários para split sem vazamento.
- Salvar matriz de confusão também para validação.
- Adicionar exportação ONNX ou TorchScript se houver necessidade de deploy.

## Padrões de projeto e boas práticas

- Separação entre dados, modelo, métricas, treino, inferência e utilitários.
- Dataclasses para transportar metadados de amostra, modelo e avaliação.
- Docstrings no padrão Google em módulos, classes e funções.
- Split por grupo para evitar vazamento entre original e augmentations.
- MLflow para rastrear parâmetros, métricas e artefatos.
- Checkpoint com `metadata` suficiente para reconstrução na inferência.
- Threshold de rejeição salvo junto do modelo.
- Early stopping por validação para reduzir treino desnecessário.
- Uso de seeds para maior reprodutibilidade.

## Relação com os outros subprojetos

O classificador depende conceitualmente do dataset produzido por `data_augmentation`, embora também possa treinar com qualquer pasta compatível. A saída prevista é usada para informar o `document_type` esperado pelo OCR/parsing.

```text
data_augmentation
      |
      v
document_classifier
      |
      v
ocr_parsing
```

## Autor

Lucas Victor Silva Pereira  
lucasvsilvap@gmail.com

## Licença

Este projeto está disponível para uso, estudo, modificação e adaptação para fins:

- acadêmicos
- educacionais
- pessoais
- institucionais não comerciais

Este projeto é licenciado sob a licença Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0).

Não é permitido o uso comercial deste projeto ou de partes dele sem autorização prévia do autor.
