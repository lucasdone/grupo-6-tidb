# SUBMISSION.md

## Time

Nome do time: **Rainbow 6**

Integrantes:
- Cristian Cunha – cristian.ccunha@gmail.com
- Victor Chang – vchang.career@gmail.com
- Marcelo Mota – marcelomota.tech@gmail.com
- Lucas Neves – lucasdedone@gmail.com
- Guilherme Campos – intergac@gmail.com

## Pitch

**Airlog.ai** monta planos de voo e logística otimizados para ferry pilots e pilotos de carga, e transforma cada trajeto planejado em uma oferta de job para pilotos freelancers.

## O que faz

O Brasil é um dos maiores fornecedores de aeronaves do mundo, mas entregar um avião (ou uma carga) entre países exige cruzar dados de aeroportos, horários de menor movimento, rotas e legislações diferentes em cada ponto da travessia. O Airlog.ai cruza a base histórica de aviação no TiDB Cloud para sugerir os melhores aeroportos, janelas de chegada/partida e rotas para o tipo de aeronave informado, usa busca vetorial para encontrar aeroportos semelhantes (perfil operacional, tráfego, tipos de aeronave que pousam lá) e gera com Amazon Bedrock uma recomendação de missão em linguagem natural. O usuário informa origem, destino, horários, aeronave e o que será entregue (carga ou a própria aeronave) e recebe um planejamento pronto, que vira uma vaga publicável para ferry pilots e tripulação de air cargo.

## Stack — marque o que você realmente usou

- [x] TiDB Cloud Starter na AWS sa-east-1
- [x] Busca vetorial no TiDB (coluna VECTOR ou EMBED_TEXT)
- [x] Amazon Bedrock (ap-southeast-1)
- [ ] Publicado na AWS -> URL no ar:
- [x] Construído com Kiro (.kiro/ commitado)

## Onde olhar

- Conexão/consultas TiDB: `src/db.py` (conexão SSL + retry, `db_cursor()`), consultas em `src/services/airport_service.py`, `src/services/route_service.py` e `src/services/mission_service.py`
- Busca vetorial: `src/services/vector_service.py` (tabela `airport_profiles` com coluna `VECTOR(1024)` e busca com `VEC_COSINE_DISTANCE`)
- Chamadas ao Bedrock: `src/services/bedrock_service.py` (Claude 3 Haiku para recomendações e Cohere embed-multilingual-v3 para embeddings, região ap-southeast-1)
- Specs e steering do Kiro: `.kiro/specs/ferryflow/` e `.kiro/steering/product.md`

## Demo

Link do vídeo de 2 minutos ou da aplicação no ar:
