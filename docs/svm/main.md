## Introdução
O modelo desenvolvido tem como objetivo prever o resultado de admissões em programas de MBA a partir de um conjunto de informações sobre os candidatos. Para isso, foi escolhida a técnica de Support Vector Machine (SVM), que é adequada para problemas de classificação por sua capacidade de encontrar limites de decisão que maximizem a separação entre as classes. Diferentemente de modelos baseados em regras, como árvores de decisão, o SVM utiliza hiperplanos e funções kernel para lidar com relações complexas entre as variáveis. Essa abordagem permite não apenas realizar previsões sobre o status de admissão, mas também capturar padrões não lineares presentes no perfil dos candidatos. Assim, o modelo funciona como ferramenta robusta de análise preditiva, indicando como combinações de características acadêmicas, demográficas e profissionais se relacionam com as chances de admissão, lista de espera ou recusa.

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
Com base na análise exploratória realizada, foram definidos e aplicados procedimentos de pré-processamento a fim de adequar os dados para a etapa de modelagem. Como o SVM é sensível à escala das variáveis e não lida diretamente com categorias, algumas etapas são essenciais para o bom desempenho da técnica.

- **Tratamento de valores ausentes**:na variável admission, valores nulos foram interpretados como recusa e substituídos por Deny. Já na variável race, valores ausentes foram preenchidos como Unknown, garantindo consistência da base.
- **Codificação de variáveis categóricas**: como o SVM opera em espaços vetoriais numéricos, colunas como gender, international, major, race, work_industry e admission foram convertidas em variáveis numéricas utilizando LabelEncoder.
- **Imputação em variáveis numéricas**: atributos como gpa, gmat e work_exp tiveram seus valores ausentes substituídos pela mediana, reduzindo o impacto de outliers.
- **Escalonamento das variáveis**: etapa fundamental para modelos SVM. As variáveis numéricas foram normalizadas utilizando StandardScaler, garantindo que todas tivessem média zero e desvio padrão igual a um. Isso evita que atributos com escalas maiores dominem o hiperplano de decisão.
- **Seleção de variáveis**: foram mantidas no conjunto de treino apenas as colunas relevantes para a análise preditiva, enquanto identificadores como application_id foram descartados.
- **Separação entre features e target**: as variáveis explicativas (X) foram extraídas das características acadêmicas, demográficas e profissionais dos candidatos, enquanto a variável alvo (y) corresponde ao status de admission.
- **Divisão em treino e teste**: o conjunto foi dividido em 80% para treino e 20% para teste, com estratificação da variável alvo para manter a proporção entre Admit, Waitlist e Deny. Essa divisão garante avaliação justa do desempenho do SVM.


=== "Base Original"
    ```python exec="1"
    --8<-- "docs/arvore-decisao/base.py"
    ```

=== "Base Preparada"
    ```python exec="1"
    --8<-- "docs/arvore-decisao/base_preparada.py"
    ```

## Divisão dos Dados
Após o pré-processamento, os dados foram divididos em dois subconjuntos: 80% para treinamento e 20% para teste. Essa proporção é amplamente utilizada em problemas de classificação e oferece quantidade suficiente de exemplos para aprendizado dos parâmetros do SVM.

A divisão foi feita utilizando a função train_test_split da biblioteca scikit-learn, com estratificação pela variável admission para manter a distribuição original das classes nos dois conjuntos. Dessa forma, o modelo SVM é treinado com um conjunto representativo da diversidade da base e posteriormente avaliado com dados nunca vistos, permitindo medir sua acurácia, robustez e capacidade de generalização.

```python exec="0"
--8<-- "docs/arvore-decisao/divisao.py"
```

## Treinamento do Modelo

=== "Modelo"
    ```python exec="on" html="1"
    --8<-- "docs/svm/svm.py"
    ```

=== "Código"
    ```python exec="0"
    --8<-- "docs/svm/svm.py"
    ```

## Avaliação do Modelo
O modelo SVM com kernel RBF apresentou uma acurácia de 83,86%, demonstrando desempenho consistente no conjunto de teste. A matriz de confusão evidencia que o classificador distingue muito bem a classe Deny, identificando corretamente praticamente todas as ocorrências dessa categoria. Esse resultado também aparece no classification report, em que a classe Deny alcança recall de 1.00 e f1-score de 0.91, indicadores que reforçam a capacidade do modelo em reconhecer padrões associados aos candidatos não admitidos.

Entretanto, o desempenho para as classes Waitlist e Admit foi significativamente inferior: o modelo não obteve recall para nenhuma delas, e a maior parte das instâncias foi absorvida pela classe Deny. Esse comportamento indica que o SVM não conseguiu distinguir adequadamente os perfis de candidatos admitidos ou colocados em lista de espera — grupos que, apesar de relevantes no processo seletivo, estão fortemente sub-representados na base.

Esse fenômeno é típico em modelos treinados sobre dados altamente desbalanceados. A predominância numérica da classe Deny faz com que o hiperplano otimizado pelo SVM priorize a separação dessa classe majoritária, resultando na subdetecção das classes minoritárias. A queda acentuada no macro average do f1-score (0.31) ilustra essa dificuldade em capturar a variabilidade entre os diferentes perfis de decisão.

## Possíveis Melhorias
1. Balanceamento das classes
A principal limitação observada decorre do forte desbalanceamento entre Deny e as demais categorias. Técnicas como SMOTE, Random Oversampling ou Class Weighting poderiam ajudar o modelo a dar maior atenção às classes menos frequentes, aumentando a capacidade de identificar candidatos Admit e Waitlist.

2. Ajuste de hiperparâmetros do SVM
A performance do modelo também pode ser aprimorada com uma busca mais ampla por hiperparâmetros, especialmente os valores de C e gamma, que controlam respectivamente a suavidade da margem e a influência dos pontos de suporte. Estratégias como GridSearchCV ou RandomizedSearchCV podem encontrar combinações que maximizem a separação entre as classes.

3. Teste de kernels alternativos
Embora o kernel RBF seja geralmente robusto, kernels como sigmoid ou poly podem capturar relações distintas entre as variáveis. Em modelos multiclasse, essa variação pode gerar hiperplanos mais adequados para distinguir grupos menores.

4. Engenharia de atributos
Criar novas combinações de variáveis (por exemplo: interação entre GMAT e GPA, senioridade profissional relativa à média da turma, agrupamento de setores semelhantes) pode ajudar o modelo a encontrar padrões mais finos que distinguem candidatos Admit e Waitlist de perfis Deny.

## Conclusão
O SVM com kernel RBF demonstrou bom desempenho geral para prever admissões em MBA, especialmente na identificação de candidatos classificados como Deny. A acurácia acima de 83% e o alto recall dessa classe evidenciam a capacidade do modelo em reconhecer o perfil predominante na base. Contudo, o desbalanceamento acentuado compromete a performance para as classes Admit e Waitlist, que praticamente não são detectadas.

Apesar dessa limitação, o estudo evidencia o potencial do SVM em problemas de classificação multiclasse, reforçando a importância de técnicas complementares — como balanceamento, tuning de hiperparâmetros e engenharia de atributos — para aprimorar sua capacidade preditiva em cenários com forte assimetria de dados.

Assim, o modelo atual fornece resultados robustos dentro das condições impostas pela base, mas abre caminho para evoluções importantes que podem torná-lo mais sensível às nuances do processo de admissão e mais justo na identificação de perfis competitivos além da classe majoritária.