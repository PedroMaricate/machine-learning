## Introdução
O algoritmo K-Means foi aplicado com o objetivo de identificar padrões e agrupar os candidatos ao MBA em clusters com características semelhantes. Diferente dos métodos supervisionados, como a árvore de decisão e o KNN, o K-Means é um algoritmo de aprendizado não supervisionado, que organiza os dados em grupos de acordo com a proximidade entre seus atributos, sem utilizar diretamente a variável de admissão como guia. Dessa forma, é possível verificar se os agrupamentos formados se aproximam das categorias reais de admissão (Admit, Waitlist e Deny), oferecendo uma perspectiva complementar sobre como os perfis acadêmicos e profissionais dos candidatos se distribuem na base de dados.

## Base de dados
A [base](https://www.kaggle.com/datasets/taweilo/mba-admission-dataset) utilizada é composta por dados sintéticos criados a partir das estatísticas da turma de 2025 do MBA de Wharton. Ela reúne informações demográficas, acadêmicas e profissionais de candidatos, como gênero, nacionalidade, área de formação, desempenho no GPA e no GMAT, além de experiência de trabalho e setor de atuação. Esses atributos foram relacionados ao status final da candidatura, categorizado como admitido, em lista de espera ou negado. Por se tratar de um conjunto de dados diversificado, é possível observar tanto os aspectos objetivos ligados ao desempenho acadêmico e profissional quanto elementos contextuais que podem influenciar o resultado do processo seletivo. Essa combinação torna o dataset especialmente relevante para análises exploratórias e para o desenvolvimento de modelos preditivos que buscam compreender os critérios implícitos de seleção em admissões de MBA.

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

## Divisão dos Dados
Diferentemente dos modelos supervisionados, em que é necessário separar os dados em conjuntos de treino e teste, o algoritmo K-Means segue uma abordagem não supervisionada. Isso significa que ele não utiliza diretamente a variável-alvo admission durante o processo de agrupamento, mas sim apenas as variáveis explicativas (gpa, gmat, work_exp, gender, international, major, race, work_industry).

Nesse contexto, todos os dados disponíveis foram utilizados para a aplicação do K-Means, de forma a identificar agrupamentos naturais dentro da base. A variável admission foi mantida apenas como referência, sendo utilizada posteriormente para comparar os clusters formados pelo modelo com as categorias reais de decisão (Admit, Waitlist e Deny). Essa estratégia permite avaliar se os grupos encontrados refletem, ao menos parcialmente, os padrões de admissão presentes no conjunto de candidatos.

```python exec="0"
--8<-- "docs/k-means/divisao_k-means.py"
```

## Treinamento do Modelo

=== "Modelo"
    ```python exec="on" html="1"    
    --8<-- "docs/k-means/treinamento_k-means.py"
    ```
=== "Código"
    ```python exec="0"    
    --8<-- "docs/k-means/treinamento_k-means.py"
    ```

## Avaliação do Modelo
O modelo K-Means foi avaliado a partir da visualização gráfica dos clusters gerados. Para representar os dados multidimensionais em duas dimensões, foi aplicada a técnica de PCA (Análise de Componentes Principais), que permite projetar as informações em um espaço 2D mantendo a maior variabilidade possível.

No gráfico, cada ponto representa um candidato ao MBA e está colorido de acordo com o cluster atribuído pelo algoritmo. Já as estrelas vermelhas indicam os centróides, ou seja, os pontos médios que definem cada grupo. A distribuição dos clusters demonstra que o modelo conseguiu separar padrões distintos dentro da base, ainda que existam áreas de sobreposição entre os grupos.

É importante ressaltar que, como se trata de uma técnica não supervisionada, o K-Means não utilizou os rótulos reais de admissão durante o treinamento. Dessa forma, a análise do gráfico permite observar o quão bem os clusters se organizaram em relação às variáveis explicativas. Apesar da presença de regiões com interseção entre cores, os resultados mostram que o algoritmo conseguiu identificar agrupamentos consistentes, refletindo diferenças entre perfis de candidatos.

Assim, o gráfico fornece uma primeira evidência visual de que o K-Means é capaz de estruturar a base em grupos coerentes, servindo como ponto de partida para avaliações quantitativas mais detalhadas por meio de métricas como silhouette score, inertia e adjusted rand index.

## Conclusão
A aplicação do algoritmo K-Means na base MBA permitiu identificar padrões e agrupar candidatos de acordo com suas características acadêmicas e profissionais. Apesar da sobreposição entre alguns clusters, a análise mostrou que o modelo conseguiu segmentar o conjunto de dados em grupos distintos, oferecendo uma visão exploratória útil sobre os perfis existentes. Dessa forma, o K-Means se mostrou uma ferramenta complementar para entender melhor a estrutura dos dados e apoiar futuras análises preditivas.