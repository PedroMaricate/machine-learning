## Introdução
O modelo de Métricas e Avaliação tem como objetivo analisar o desempenho dos algoritmos aplicados à base de admissões de MBA, fornecendo medidas quantitativas que permitem interpretar a qualidade das previsões realizadas. Para isso, são utilizadas métricas clássicas de classificação, como acurácia, precisão, recall e F1-Score, além da matriz de confusão, que detalha os acertos e erros por classe. Esses indicadores são essenciais para identificar pontos fortes e limitações do modelo, especialmente em bases que apresentam certo desbalanceamento entre as categorias de admissão. Dessa forma, o uso das métricas garante uma avaliação mais completa e confiável, servindo de apoio para a comparação entre diferentes algoritmos já aplicados, como Árvore de Decisão, KNN e K-Means.

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
Após o pré-processamento, os dados foram divididos em dois subconjuntos: 80% para treinamento e 20% para teste. Essa divisão é amplamente utilizada em problemas de classificação, pois garante exemplos suficientes para o ajuste dos algoritmos e, ao mesmo tempo, reserva uma amostra representativa para a etapa de avaliação.

A separação foi realizada por meio da função train_test_split da biblioteca scikit-learn, utilizando a opção de estratificação pela variável alvo (admission). Esse procedimento assegura que a proporção original entre as classes (Admit, Waitlist e Deny) seja mantida em ambos os subconjuntos, evitando que o modelo seja influenciado por distribuições desbalanceadas.

Com isso, o conjunto de treinamento é destinado ao ajuste dos algoritmos testados, enquanto o conjunto de teste serve como base imparcial para a aplicação das métricas de desempenho, como acurácia, precisão, recall, F1-Score e matriz de confusão, que permitem avaliar a qualidade das previsões obtidas.

```python exec="0"
--8<-- "docs/arvore-decisao/divisao.py"
```

## Treinamento do Modelo
=== "Código"
    ```python exec="0"    
    --8<-- "docs/metricas-avaliacao/treinamento-metricas.py"
    ```

## Avaliação do Modelo
== Avaliação KNN

    O modelo KNN apresentou acurácia de aproximadamente 84%, o que indica um bom desempenho geral na classificação dos candidatos. Pela matriz de confusão, nota-se que o modelo classifica corretamente a maior parte da classe Deny (0), mas apresenta dificuldades em distinguir as classes Waitlist (1) e Admit (2), refletido nos valores baixos de precision e recall para essas categorias. Isso é esperado, já que as classes estão desbalanceadas. O weighted F1-score de 0.81 reforça que, embora haja desequilíbrio, o modelo consegue manter uma performance satisfatória no conjunto de teste.

== Avaliação K-Means

    No caso do K-Means, o desempenho foi mais limitado. O Silhouette Score (0,13) mostra que os clusters formados não são bem separados, indicando sobreposição entre os grupos. Além disso, o Adjusted Rand Index (0,04) confirma que os clusters obtidos apresentam baixa concordância com as classes reais de admissão (Deny, Waitlist e Admit). A matriz de confusão evidencia que os clusters não correspondem claramente às categorias originais, reforçando que o K-Means não captura adequadamente os padrões da base para fins de classificação supervisionada.