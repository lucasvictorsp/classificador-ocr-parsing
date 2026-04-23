# Document AI SaaS – Classificação, OCR e Parsing de Documentos

## Visão Geral

```text
Este projeto apresenta uma arquitetura completa e escalável para processamento automatizado de documentos brasileiros (CNH, RG e CPF), combinando técnicas de visão computacional, OCR e processamento de linguagem, incluindo:
```

- Classificação de documento
- Extração de texto (OCR)
- Parsing estruturado dos dados
- Retorno assíncrono ao cliente

O sistema foi concebido como um SaaS altamente escalável, orientado a throughput, utilizando mensageria e processamento assíncrono. Concebido para simular um ambiente real de produção, priorizando escalabilidade, eficiência operacional e robustez frente a dados não controlados provenientes do mundo real.

---

## Arquitetura Cloud-Agnostic

A arquitetura foi personificada na **Google Cloud Platform (GCP)**, devido à maior afinidade e experiência do autor com esse ambiente.

No entanto, o projeto é completamente **agnóstico a cloud**, podendo ser executado em:

- AWS  
- Azure  
- Kubernetes on-premise  
- Outras clouds privadas  

---

## Arquitetura do Sistema

![Arquitetura do Projeto](arquitetura_do_projeto/desenho_arquitetura_do_projeto.png)

### Fluxo do Sistema

#### 1. Upload do Documento

Cliente envia imagem (**máximo 8MB**)

A API valida:
- Tipo de arquivo  
- Integridade  

Após upload completo:

- Geração de `job_id`  
- Cálculo de hash do arquivo (**deduplicação**)  
- Publicação em sistema de mensageria  

**Importante:**  
A mensagem só é enviada após o upload completo, evitando leitura prematura por workers.

---

#### 2. Segurança

Worker isolado executa:

- Antivírus baseado em análise de bytes (**sem abrir o arquivo**)  
- Validação de integridade  

**Resultado:**

- Arquivo suspeito → Bucket de quarentena  
- Arquivo seguro → Bucket seguro  

---

#### 3. Processamento Assíncrono

O sistema é orientado a eventos:

- Cliente recebe `job_id` imediatamente  
- Processamento ocorre de forma assíncrona  

**Estados possíveis:**

- Uploaded  
- Processing  
- Quarantine  
- Waiting Human Review  
- Completed  

---

#### 4. Classificação

- **Modelo:** EfficientNet-B0  
- **Técnica:** Transfer Learning  

Na POC:

- Fine-tuning leve nas camadas finais  

**Fluxo:**

- Alta confiança → segue pipeline  
- Baixa confiança → fallback com LLM  

---

#### 5. OCR e Parsing

##### Pipeline Principal

- OCR utilizando **Tesseract** (execução local)  
- Parsing com **Regex**  

---

##### Decisão Técnica

Para a etapa de OCR, foi utilizado o **Tesseract** em ambiente local.

A escolha foi baseada em um trade-off importante entre:

- Tempo de resposta  
- Custo computacional  
- Complexidade operacional  

Modelos mais robustos como:

- PaddleOCR  
- docTR  

podem oferecer maior acurácia em alguns cenários, porém:

- Possuem maior custo computacional  
- Introduzem maior latência  
- Exigem maior esforço operacional  

---

##### Trade-off do Pipeline

O projeto foi desenhado com a seguinte estratégia:

- **OCR + Regex** como pipeline principal  
- **LLM** como fallback  

###### Motivação

- OCR + Regex → rápido, barato e determinístico  
- LLM → mais robusto, porém mais caro e mais lento  

Portanto:

> O OCR combinado com Regex atende a maior parte dos casos com eficiência operacional, enquanto o uso de LLM é reservado apenas para cenários onde a confiança do pipeline principal é baixa.

---

##### Estratégia de Produção

Em um ambiente produtivo em cloud, como o GCP, uma alternativa viável seria o uso do:

- Vision API  

Esse serviço oferece:

- Alta acurácia  
- Escalabilidade gerenciada  
- Integração nativa com outros serviços cloud  

---

##### Fine-tuning do OCR

Uma evolução importante do projeto é a possibilidade de realizar fine-tuning do modelo de OCR, incluindo:

- Adaptação e customização do Tesseract para o domínio
- Possibilidade de fine-tuning em modelos de OCR. No caso do Tesseract, esse processo não é trivial, porém modelos como *PaddleOCR* permitem essa customização de forma mais direta.
- Ajuste para documentos brasileiros (CNH, RG, CPF)  
- Melhoria de reconhecimento de padrões específicos (nomes, números, datas)  

Essa abordagem permite:

- Aumentar a acurácia  
- Manter baixo custo computacional  
- Preservar baixa latência  

---

##### Parsing

Após a etapa de OCR:

- Os dados são estruturados utilizando Regex  
- Regras específicas são aplicadas por tipo de documento  

##### Exemplos

- CPF → validação por dígito verificador  
- Datas → normalização de formato  
- Nomes → extração baseada em contexto  

---

#### 6. Fallback Inteligente

Utilizado quando:

- Classificação falha
- OCR falha 
- Parsing falha  

- O sistema aciona um modelo LLM  
- O LLM realiza classificação, OCR e/ou parsing mais robusto  
- Caso ainda haja baixa confiança → encaminhamento para análise humana  

---

#### 7. Human-in-the-loop

- Documentos críticos são enviados para revisão humana  
- Resultado alimenta melhoria contínua do sistema  

---

#### 8. Aprendizado Contínuo

Documentos processados via:

- LLM  
- Revisão humana  

São utilizados para:

- Fine-tuning futuro  
- Evolução dos modelos  

---

### Avaliação de Confiança

A tomada de decisão do pipeline é baseada em métricas de confiança, tais como:

- Probabilidade softmax (classificação)
- Score de OCR
- Validação de parsing (ex: CPF válido)

Thresholds são definidos empiricamente e podem ser ajustados via monitoramento em produção.

---

### ⚡ Estratégia de Otimização

O pipeline segue a seguinte lógica:

- Priorizar soluções determinísticas e baratas  
- Utilizar LLM apenas quando necessário  
- Minimizar custo por requisição  
- Garantir baixa latência média  

O objetivo é manter o tempo de resposta médio baixo (baixa latência) e o custo por requisição reduzido, mesmo sob alta carga.

---

### 💡 Insight Arquitetural

A combinação de **OCR + Regex** como primeira camada e **LLM como fallback** garante um equilíbrio eficiente entre custo, performance e qualidade.

---

## Observabilidade

O sistema deve ser totalmente observável, incluindo:

- Logs estruturados (por `job_id`)  
- Métricas de throughput  
- Tempo por etapa (classificação, OCR, parsing)  
- Taxa de fallback para LLM  
- Taxa de erro por etapa

A observabilidade é fundamental para garantir confiabilidade em produção e permitir evolução contínua do sistema.

---

### Benefícios

Isso permite:

- Debug eficiente  
- Monitoramento de performance  
- Ajuste de thresholds  

---

## ⚡ Arquitetura Orientada a Throughput

O sistema foi projetado para suportar alta volumetria através de:

- Processamento assíncrono  
- Workers desacoplados  
- Sistema de mensageria como buffer (backpressure)  
- Escalabilidade horizontal via containers

Essa abordagem permite que o sistema escale horizontalmente de forma eficiente, suportando crescimento de carga sem necessidade de reestruturação da arquitetura.

Essa arquitetura permite desacoplamento completo entre ingestão e processamento, aumentando resiliência e tolerância a falhas.

---

### Benefícios

Isso permite que o sistema absorva picos de carga sem degradação imediata.

---

---

## Alternativa SaaS: Google Document AI

Uma alternativa ao pipeline desenvolvido neste projeto é a utilização do serviço totalmente gerenciado **Google Document AI**.

Esse serviço permite:

- Classificação automática de documentos  
- Extração estruturada de dados (parsing)  
- Uso de modelos pré-treinados para documentos diversos  
- Possibilidade de **fine-tuning** para casos específicos  

---

### Capacidades do Document AI

Com o Document AI, é possível:

- Criar modelos customizados de classificação de documentos  
- Utilizar parsers pré-treinados do Google (ex: documentos de identidade, formulários, invoices, etc.)  
- Realizar fine-tuning dos modelos para melhorar a acurácia em domínios específicos  
- Reduzir significativamente o esforço de implementação de OCR + parsing  

---

### Trade-off Técnico e Financeiro

Apesar das vantagens, essa abordagem envolve um trade-off importante:

### Benefícios

- Alta acurácia  
- Infraestrutura totalmente gerenciada  
- Redução de complexidade operacional  
- Escalabilidade nativa  

### Pontos de atenção

- Custo baseado em:
  - Número de chamadas (requests)  
  - Tamanho/volume dos documentos processados  
- Dependência de vendor (vendor lock-in)  
- Menor controle sobre o pipeline interno  

---

### Considerações de FinOps

Antes de adotar essa solução em produção, é fundamental realizar um estudo de viabilidade financeira, incluindo:

- Custo por documento processado  
- Volume esperado de requisições  
- Comparação com custo do pipeline próprio (OCR + Regex + LLM)  
- Análise de custo vs. ganho de acurácia 

---

### Insight Arquitetural

Uma abordagem híbrida também pode ser considerada:

- Pipeline próprio como padrão (baixo custo)  
- Document AI como fallback para casos complexos.  

Essa estratégia permite equilibrar **custo, performance e qualidade**, mantendo controle operacional e aproveitando serviços gerenciados quando necessário.

Novamente, aqui é necessário estudo de *FinOps*, *quailidade* e *tempo de resposta* para verificar qual processo de fallback é mais vantajoso, com *Document AI* ou *LLM*.

---

## Partes Desenvolvidas no Projeto

Devido à extensão do projeto, as seguintes partes foram implementadas na prática:

- Treinamento da CNN para classificação de documentos  
- OCR  
- Parsing dos dados  

Outras partes foram projetadas arquiteturalmente.

---

## Estrutura Atual do Projeto

Devido ao escopo e à extensão do projeto, o desenvolvimento foi realizado de forma **modular**, priorizando a qualidade técnica de cada componente individual.
Atualmente, o projeto **não possui uma integração completa** entre todos os módulos implementados.
As principais partes desenvolvidas estão organizadas nas seguintes pastas:

- `src/data_augmentation`  
- `src/document_classifier`  
- `src/ocr_parsing`  

---

### Organização dos Módulos

Cada um desses módulos foi desenvolvido de forma independente, contendo:

- Implementação completa da funcionalidade proposta  
- Scripts de execução  
- Documentação específica  
- Avaliação de resultados (quando aplicável)  

---

### Como Navegar no Projeto

Para entender completamente o funcionamento do sistema, é necessário acessar individualmente cada módulo:

#### `src/data_augmentation`

- Pipeline completo de aumento de dados  
- Estratégias para simulação de dados reais  
- Configurações e cenários aplicados  

---

#### `src/document_classifier`

- Treinamento da CNN (EfficientNet-B0)  
- Processo de fine-tuning  
- Métricas de avaliação do modelo  
- Scripts de treino e validação  

---

#### `src/ocr_parsing`

- Pipeline de OCR utilizando Tesseract  
- Regras de parsing baseadas em Regex  
- Avaliação da qualidade da extração de dados  

---

### Observação Importante

Como o foco do projeto foi demonstrar a construção de cada etapa do pipeline de forma sólida e independente, a integração entre os módulos **não foi finalizada nesta versão**.

Essa integração envolveria:

- Orquestração entre classificação, OCR e parsing  
- Padronização de inputs e outputs entre módulos  
- Pipeline end-to-end automatizado  

---

### Próximos Passos Naturais

A evolução natural do projeto inclui:

- Integração completa entre os módulos  
- Criação de um pipeline unificado  
- Deploy end-to-end em ambiente cloud  
- Monitoramento integrado de todas as etapas  

---

## Treinamento da CNN

### Dataset Utilizado

Dataset público disponível no repositório de [Ricardo Neves Junior](https://github.com/ricardobnjunior/Brazilian-Identity-Document-Dataset).


```bibtex
@inproceedings{sibgrapi_estendido,
 author = {Álysson Soares and Ricardo das Neves Junior and Byron Bezerra},
 title = {BID Dataset: a challenge dataset for document processing tasks},
 booktitle = {Anais Estendidos do XXXIII Conference on Graphics, Patterns and Images},
 location = {Evento Online},
 year = {2020},
 pages = {143--146},
 publisher = {SBC},
 doi = {10.5753/sibgrapi.est.2020.12997}
}
```

---

###  Problema do Dataset

- Imagens perfeitas  
- Sem fundo  
- Sem ruído  
- Fora do padrão real de produção  

---

## Data Augmentation

Para resolver o problema de generalização, foi aplicado um pipeline avançado de data augmentation.
O objetivo principal do data augmentation não é apenas aumentar o dataset, mas sim aproximar a distribuição dos dados de treino das condições reais de produção.

### Proporção

- 1 imagem real → 1 imagem aumentada  
- Proporção: **50%**  

**Total por documento:**

- 200 imagens reais (100 frente + 100 verso)  
- 200 imagens aumentadas  

**Total por tipo:** 400 imagens  
**Total geral:** 1200 imagens  

---

### Técnicas de Data Augmentation Aplicadas

Baseado no pipeline real implementado:

#### Geometria

- Rotação  
- Affine transformations  
- Shear  
- Translação  
- Perspective transform  

#### Iluminação

- Brightness e contrast  
- Hue e saturation  
- Gamma  
- Random shadow  

#### Simulação de câmera

- Motion blur  
- Gaussian blur  
- Defocus blur  
- Ruído gaussiano  

#### Compressão

- JPEG compression com qualidade variável  

#### Fundo (muito importante)

- Substituição por fundos coloridos realistas  
- Simulação de documentos fotografados  

---

### Cenários Compostos (Diferencial)

O pipeline não aplica transformações isoladas, mas cenários realistas:

- Inclinação + fundo + iluminação  
- Perspectiva + sombra  
- Baixa luz + blur  
- Ruído + compressão  
- Distorção forte + fundo  
- Degradação múltipla  

---

### Controle de Data Leakage (Fuga de Informação)

Um ponto crítico no treinamento do modelo é evitar **data leakage**, ou seja, fuga de informação entre os conjuntos de treino, validação e teste.

No contexto deste projeto, isso é especialmente importante devido ao uso intensivo de data augmentation.

---

#### Problema

Se uma imagem original for utilizada no conjunto de treino e uma versão aumentada dessa mesma imagem for utilizada no conjunto de teste ou validação, o modelo pode:

- Memorizar padrões específicos do documento  
- Obter métricas artificialmente infladas  
- Perder capacidade de generalização  

---

#### Solução Implementada

Para evitar esse problema, foi adotada a seguinte estratégia:

- Cada documento possui um identificador único  
- Todas as suas variações (original + aumentadas) são tratadas como um único grupo  
- O split dos dados é feito com base no identificador do documento  

---

#### Regra de Split

O particionamento do dataset segue a lógica:

- **Treino:** conjunto de documentos completos (original + augmentations)  
- **Validação:** conjunto distinto de documentos completos  
- **Teste:** conjunto distinto de documentos completos  

Ou seja:

> A imagem original e todas as suas versões aumentadas **sempre permanecem no mesmo conjunto**.

---

#### Benefícios

- Evita vazamento de informação entre datasets  
- Garante avaliação realista do modelo  
- Preserva a integridade estatística do experimento  
- Melhora a capacidade de generalização em produção  

---

#### Insight

Esse cuidado é essencial em pipelines que utilizam data augmentation intensivo, sendo uma prática obrigatória em sistemas de visão computacional aplicados a dados sensíveis como documentos.

---

#### Sincronização Completa

Transformações aplicadas simultaneamente em:

- Imagem  
- Máscara  
- Bounding boxes  

Com uso de `ReplayCompose`.

---

#### Rastreabilidade

Cada transformação aplicada gera um registro contendo:

- Nome da transformação  
- Parâmetros  
- Targets afetados (imagem, máscara, annotations)  

---

### Monitoramento do Treinamento

O processo de treinamento deve ser monitorado continuamente.

Isso inclui:

- Loss  
- Accuracy  
- F1-score  
- Overfitting  
- Drift de dados  
- Comparação entre experimentos  

#### Ferramentas recomendadas

- MLflow  
- Weights & Biases  

---

## LGPD e Segurança

### Proteção de Dados

- Buckets temporários para imagens  
- Exclusão automática após processamento
- Possibilidade de anonimização irreversível dos dados
- Separação entre dados sensíveis e dados de auditoria

### Persistência

Banco armazena apenas:

- Dados anonimizados e desacoplados de qualquer informação sensível identificável
- Logs de auditoria  

### Princípios LGPD

- Minimização de dados  
- Retenção controlada  
- Segurança por design  

---

## Escalabilidade

- Arquitetura baseada em mensageria  
- Processamento assíncrono  
- Workers desacoplados  
- Auto scaling  

---

## Diferenciais

- Arquitetura orientada a eventos  
- Pipeline resiliente com fallback  
- Human-in-the-loop  
- Data augmentation realista  
- Rastreabilidade completa  
- Preparado para MLOps  

---

## Conclusão

Este projeto demonstra a construção de um sistema de Document AI orientado a produção, equilibrando:

- Escalabilidade
- Eficiência operacional
- Custo computacional
- Robustez frente a dados reais

A arquitetura proposta permite evolução contínua, sendo preparada para ambientes de produção com alta volumetria, requisitos rigorosos de segurança e governança de dados.

---

## Uso de LLM no Desenvolvimento

Este projeto foi desenvolvido com o auxílio de modelos de linguagem de larga escala (LLMs), em especial o modelo Gemini.

O uso de LLMs contribuiu em diferentes aspectos do projeto, incluindo:

- Suporte na implementação de código  
- Apoio na escrita e organização da documentação  
- Estruturação do `README.md`  
- Sugestões de melhorias na arquitetura  
- Discussão de trade-offs técnicos  

---

### Papel do LLM no Projeto

O modelo foi utilizado como uma ferramenta de apoio ao desenvolvimento, atuando como:

- Assistente técnico  
- Revisor de código e documentação  
- Fonte de sugestões para boas práticas  

---

### Responsabilidade Técnica

Apesar do uso de LLMs:

- Todas as decisões finais de arquitetura foram avaliadas criticamente  
- As escolhas técnicas foram baseadas em conhecimento prévio, experiência prática e validação dos trade-offs  
- O projeto reflete a visão e responsabilidade técnica do autor  

---

### Consideração Final

O uso consciente de LLMs foi tratado como parte do processo de engenharia, mantendo o foco em qualidade, robustez e entendimento profundo das soluções adotadas.

---

## Ambiente de Desenvolvimento e Execução

Este projeto foi desenvolvido e testado em:

- Windows 10  
- Python 3.12  

Apesar disso, o ambiente local de desenvolvimento não limita sua execução em produção, pois o projeto foi estruturado com foco em **portabilidade e reprodutibilidade**, utilizando:

- Docker  
- Poetry  
- `pyproject.toml`  
- `poetry.lock`  

Essa abordagem garante que diferenças entre o ambiente de desenvolvimento no Windows e um ambiente robusto de execução em servidores Linux não impactem a aplicação do projeto, já que as dependências, versões e comportamento da aplicação ficam padronizados via containerização e gerenciamento declarativo de pacotes.

### 💡 Em outras palavras

- O Windows foi apenas o ambiente de desenvolvimento  
- O runtime do projeto é desacoplado do sistema operacional do autor  
- O deploy em ambientes Linux permanece consistente, previsível e reprodutível  

---

## Autor

**Lucas Victor Silva Pereira**
[lucasvsilvap@gmail.com](mailto:lucasvsilvap@gmail.com)

---

## Licença

Este projeto está disponível para uso, estudo, modificação e adaptação para fins:

- acadêmicos  
- educacionais  
- pessoais  
- institucionais não comerciais
- Este projeto é licenciado sob a licença *Creative Commons Attribution-NonCommercial 4.0* (**CC BY-NC 4.0**).

❗ **Não é permitido o uso comercial deste projeto ou de partes dele sem autorização prévia do autor.**