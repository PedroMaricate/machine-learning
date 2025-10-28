## Introdução
O modelo desenvolvido tem como objetivo prever o resultado de admissões em programas de MBA a partir de um conjunto de informações sobre os candidatos. Para isso, foi escolhida a técnica de Random Forest, um método de aprendizado de máquina baseado em um conjunto de árvores de decisão que, ao trabalharem de forma conjunta, proporcionam maior precisão e robustez nas previsões. Essa abordagem permite não apenas identificar o status de admissão de cada candidato, mas também compreender a relevância de diferentes variáveis no processo seletivo, por meio da análise da importância das features. Assim, o modelo combina poder preditivo e capacidade interpretativa, funcionando tanto como uma ferramenta de classificação quanto de apoio à decisão, ao destacar os fatores mais determinantes no perfil de um candidato admitido.

## Base de dados
A [base](https://www.kaggle.com/datasets/taweilo/mba-admission-dataset) utilizada é composta por dados sintéticos criados a partir das estatísticas da turma de 2025 do MBA de Wharton. Ela reúne informações demográficas, acadêmicas e profissionais de candidatos, como gênero, nacionalidade, área de formação, desempenho no GPA e no GMAT, além de experiência de trabalho e setor de atuação. Esses atributos foram relacionados ao status final da candidatura, categorizado como admitido, em lista de espera ou negado. Por se tratar de um conjunto de dados diversificado, é possível observar tanto os aspectos objetivos ligados ao desempenho acadêmico e profissional quanto elementos contextuais que podem influenciar o resultado do processo seletivo. Essa combinação torna o dataset especialmente relevante para análises exploratórias e para o desenvolvimento de modelos preditivos que buscam compreender os critérios implícitos de seleção em admissões de MBA.

## Exploração dos Dados
A seguir foi realizada uma análise exploratória da base de dados, com o objetivo de compreender o significado e a composição de cada coluna. Essa etapa busca identificar possíveis problemas, como valores ausentes ou distribuições desbalanceadas, que podem influenciar diretamente a qualidade do modelo. As visualizações e estatísticas descritivas permitem observar padrões, tendências e discrepâncias entre os candidatos, fornecendo subsídios importantes para orientar as decisões de pré-processamento e a construção da árvore de decisão.

=== "gender"
    A variável gênero apresenta uma diferença significativa na quantidade de aplicações. Observa-se uma predominância de candidatos do sexo masculino em comparação às candidatas do sexo feminino, o que indica uma distribuição desigual nesse aspecto. Essa discrepância pode refletir tendências do mercado de MBA ou características específicas da base gerada. Além disso, é um fator importante a ser considerado no modelo, já que possíveis vieses de gênero podem influenciar tanto a análise quanto as previsões de admissão.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/gender.py"
    ```

=== "international"
    A variável alunos internacionais mostra que a maior parte das aplicações é de candidatos domésticos (não internacionais), enquanto os estudantes internacionais representam uma parcela menor do total. Essa diferença pode indicar que os programas de MBA ainda têm maior procura local, embora o número de aplicações internacionais seja relevante para demonstrar a diversidade e a atratividade global da instituição. Essa característica pode influenciar o modelo de predição, visto que fatores como origem do aluno podem estar relacionados às taxas de aceitação.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/international.py"
    ```

=== "gpa"
    A variável GPA apresenta distribuição concentrada em torno de valores relativamente altos, entre 3.1 e 3.3, o que indica que a maior parte dos candidatos possui desempenho acadêmico consistente. A mediana situa-se pouco acima de 3.2, reforçando esse padrão. Observa-se ainda a presença de alguns valores atípicos, tanto abaixo de 2.8 quanto acima de 3.6, que representam candidatos com desempenho fora do perfil predominante. Esses outliers, embora pouco frequentes, podem influenciar a análise estatística e devem ser considerados no pré-processamento ou na interpretação dos resultados do modelo. De forma geral, a distribuição do GPA sugere que a base é composta majoritariamente por candidatos academicamente fortes, o que pode ser um dos critérios determinantes no processo de admissão.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/gpa.py"
    ```

=== "major"
    A variável major, que representa a área de formação acadêmica dos candidatos, apresenta distribuição relativamente equilibrada entre as categorias, mas com destaque para Humanidades, que concentra o maior número de aplicações. As áreas de STEM e Business aparecem em proporções semelhantes, ambas com menor participação em relação a Humanidades. Essa diferença pode refletir o perfil da amostra, indicando maior procura de candidatos oriundos de cursos de Humanidades pelos programas de MBA. A análise dessa variável é relevante para verificar se determinadas formações acadêmicas têm maior representatividade ou desempenham papel diferenciado nos resultados de admissão.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/major.py"
    ```

=== "race"
    A variável raça apresenta distribuição diversificada entre os candidatos, com destaque para a categoria de pessoas que preferiram não se identificar, seguida pelo grupo White. Em seguida aparecem Asian, Black e Hispanic, enquanto a categoria Other concentra a menor quantidade de aplicações. Essa composição evidencia tanto a representatividade de diferentes origens raciais quanto a limitação do campo para candidatos internacionais. A análise dessa variável é importante para compreender a diversidade do conjunto de dados e avaliar se há possíveis diferenças de perfil que podem influenciar nos resultados de admissão.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/race.py"
    ```    

=== "gmat"
    A variável GMAT apresenta uma distribuição concentrada entre 600 e 700 pontos, faixa onde se encontra a maior parte dos candidatos. O pico de frequência ocorre próximo de 650 pontos, o que sugere que esse valor é representativo do desempenho médio dos aplicantes. Apesar dessa concentração, também há candidatos com pontuações mais baixas, em torno de 570, bem como outros que alcançam notas elevadas acima de 750, embora em menor quantidade. Essa distribuição indica que, em geral, os candidatos possuem desempenho sólido no exame, mas com variação suficiente para permitir que o modelo identifique padrões relacionados ao status de admissão.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/gmat.py"
    ``` 
=== "work_exp"
    A variável experiência profissional apresenta distribuição concentrada entre 4 e 6 anos de atuação no mercado, com destaque para os candidatos que possuem 5 anos de experiência, que representam a maior parte das aplicações. Os extremos da distribuição, com candidatos que possuem apenas 1 ou 2 anos de experiência e aqueles com mais de 7 anos, aparecem em menor número, configurando perfis menos frequentes na amostra. Esse padrão sugere que a base de dados está composta principalmente por profissionais em estágio intermediário de carreira, o que reflete o perfil típico de aplicantes a programas de MBA. Essa variável é particularmente relevante, pois pode influenciar diretamente nas chances de admissão, uma vez que a experiência prática é um critério valorizado nas seleções.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/work_exp.py"
    ``` 

=== "work_industry"
    A variável setor de experiência profissional revela que a maior parte dos candidatos possui trajetória em Consultoria, que se destaca amplamente em relação aos demais setores. Em seguida aparecem PE/VC (Private Equity e Venture Capital), Tecnologia e setores ligados ao serviço público ou organizações sem fins lucrativos, todos com participação significativa. Áreas tradicionais como Investment Banking e Financial Services também se mostram relevantes, mas em menor proporção. Já setores como Saúde, Bens de Consumo (CPG), Mídia/Entretenimento, Varejo, Imobiliário e Energia aparecem de forma mais restrita, representando nichos específicos da amostra. Essa distribuição indica que o MBA atrai predominantemente profissionais de consultoria e finanças, mas também apresenta diversidade ao incluir candidatos de áreas emergentes e de setores menos tradicionais.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/work_industry.py"
    ```

## Pré-Processamento 
Com base na análise exploratória realizada, foram definidos e aplicados procedimentos de pré-processamento a fim de adequar os dados para a etapa de modelagem.
As principais etapas conduzidas incluem:

- **Tratamento de valores ausentes**:na variável admission, valores nulos foram interpretados como recusa e substituídos por Deny. Já na variável race, os valores ausentes foram preenchidos como Unknown, garantindo a consistência da base.
- **Codificação de variáveis categóricas**: colunas como gender, international, major, race, work_industry e admission foram convertidas em variáveis numéricas por meio do método LabelEncoder, possibilitando sua utilização pelo modelo de árvore de decisão.
- **Imputação em variáveis numéricas**: atributos como gpa, gmat e work_exp tiveram seus valores ausentes substituídos pela mediana, minimizando a influência de outliers e preservando a distribuição original dos dados.
- **Seleção de variáveis**: foram mantidas no conjunto de treino apenas as colunas relevantes para a análise preditiva, enquanto identificadores como application_id foram descartados por não possuírem valor analítico.
- **Separação entre features e target**: as variáveis explicativas (X) foram definidas a partir das características acadêmicas, demográficas e profissionais dos candidatos, enquanto a variável alvo (y) corresponde ao status de admission.
- **Divisão em treino e teste**: o conjunto de dados foi dividido em duas partes, com 80% para treino e 20% para teste, garantindo estratificação da variável alvo para preservar a proporção entre as classes.


=== "Base Original"
    ```python exec="1"
    --8<-- "docs/arvore-decisao/base.py"
    ```

=== "Base Preparada"
    ```python exec="1"
    --8<-- "docs/arvore-decisao/base_preparada.py"
    ```

## Divisão dos dados
Nesta etapa, o conjunto de dados foi dividido em duas partes: treinamento e teste.
O conjunto de treinamento é utilizado para ajustar o modelo de Random Forest, permitindo que ele aprenda os padrões que relacionam as variáveis de entrada com o resultado de admissão.
Já o conjunto de teste é usado para avaliar o desempenho do modelo em dados nunca vistos, garantindo que a análise reflita a capacidade real de generalização.

A proporção de 80% para treino e 20% para teste foi adotada, e a divisão foi feita de forma estratificada, mantendo a proporção entre candidatos admitidos e não admitidos. Essa estratégia evita que o modelo aprenda de forma enviesada e assegura maior consistência na avaliação final.

```python exec="0"
--8<-- "docs/random-forest/divisao_random.py"
```
## Treinamento do Modelo
=== "Modelo"
    ```python exec="on" html="1"    
    --8<-- "docs/random-forest/treinamento_random.py"
    ```
=== "Código"
    ```python exec="0"    
    --8<-- "docs/random-forest/treinamento_random.py"
    ```

## Avaliação do Modelo
O modelo de Random Forest apresentou um desempenho geral consistente, atingindo uma acurácia de 85,39% nos dados de teste e um OOB score de 84,88%, o que indica boa capacidade de generalização — ou seja, o modelo mantém resultados estáveis mesmo em amostras fora do treinamento.

A matriz de classificação mostra que o modelo tem excelente desempenho na classe 0 (não admitidos), com precisão de 87,8% e recall de 96,3%, resultando em um f1-score de 0,918.
Já para a classe 1 (admitidos), o modelo apresenta uma precisão menor (49,3%) e recall de 21,1%, o que indica dificuldade em identificar todos os candidatos realmente admitidos — uma consequência comum em bases desbalanceadas, onde há muito mais casos de não admissão do que de admissão.

Mesmo com essa limitação, o modelo demonstra boa capacidade de separar os dois grupos, sendo mais conservador ao prever admissões (o que reduz falsos positivos, mas aumenta falsos negativos).

### Importância das Variáveis

Abaixo estão as variáveis que mais contribuíram para as decisões do modelo Random Forest, segundo o método **Mean Decrease in Impurity (MDI)**:

| Ranking | Variável                | Importância |
|:--------:|:------------------------|-------------:|
| 🥇 1 | **GPA (média acadêmica)**          | **0.2970** |
| 🥈 2 | **GMAT (nota do exame)**          | **0.2827** |
| 🥉 3 | **Experiência profissional (work_exp)** | **0.0986** |
| 4 | Área de formação – *Humanities* | 0.0339 |
| 5 | Área de formação – *STEM* | 0.0312 |
| 6 | Gênero – Masculino | 0.0285 |
| 7 | Setor – Consulting | 0.0224 |
| 8 | Raça – White | 0.0206 |
| 9 | Setor – Private Equity / VC | 0.0176 |
| 10 | Setor – Technology | 0.0168 |

*Observação:*  
O modelo dá maior peso ao **desempenho acadêmico e profissional**, reforçando que candidatos com **GPA** e **GMAT** altos e **maior experiência de trabalho** têm maior probabilidade de admissão.

