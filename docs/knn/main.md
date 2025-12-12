## Introdução
O algoritmo K-Nearest Neighbors (KNN) foi utilizado como alternativa para prever o status de admissão dos candidatos ao MBA. Esse método classifica cada candidato com base nos perfis mais semelhantes do conjunto de treino, considerando atributos como GPA, GMAT e experiência profissional. Por sua simplicidade e flexibilidade, o KNN não depende de relações lineares e oferece uma visão baseada em proximidade entre os candidatos, funcionando como complemento às análises realizadas com a árvore de decisão.

## Base de dados
A [base](https://www.kaggle.com/datasets/taweilo/mba-admission-dataset) utilizada é composta por dados sintéticos criados a partir das estatísticas da turma de 2025 do MBA de Wharton. Ela reúne informações demográficas, acadêmicas e profissionais de candidatos, como gênero, nacionalidade, área de formação, desempenho no GPA e no GMAT, além de experiência de trabalho e setor de atuação. Esses atributos foram relacionados ao status final da candidatura, categorizado como admitido, em lista de espera ou negado. Por se tratar de um conjunto de dados diversificado, é possível observar tanto os aspectos objetivos ligados ao desempenho acadêmico e profissional quanto elementos contextuais que podem influenciar o resultado do processo seletivo. Essa combinação torna o dataset especialmente relevante para análises exploratórias e para o desenvolvimento de modelos preditivos que buscam compreender os critérios implícitos de seleção em admissões de MBA.

=== "gender"
    A variável gênero apresenta uma diferença significativa na quantidade de aplicações. Observa-se uma predominância de candidatos do sexo masculino em comparação às candidatas do sexo feminino, o que indica uma distribuição desigual nesse aspecto. Essa discrepância pode refletir tendências do mercado de MBA ou características específicas da base gerada. Além disso, é um fator importante a ser considerado no modelo, já que possíveis vieses de gênero podem influenciar tanto a análise quanto as previsões de admissão.

    ![Frequência de gênero por aplicações](./arvore-decisao/img/frequencia_genero_aplicacoes.png)

=== "international"
    A variável alunos internacionais mostra que a maior parte das aplicações é de candidatos domésticos (não internacionais), enquanto os estudantes internacionais representam uma parcela menor do total. Essa diferença pode indicar que os programas de MBA ainda têm maior procura local, embora o número de aplicações internacionais seja relevante para demonstrar a diversidade e a atratividade global da instituição. Essa característica pode influenciar o modelo de predição, visto que fatores como origem do aluno podem estar relacionados às taxas de aceitação.

    ![Frequência de alunos internacionais por aplicações](docs/arvore-decisao/img/frequencia_alunos_internacionais.png)

=== "gpa"
    A variável GPA apresenta distribuição concentrada em torno de valores relativamente altos, entre 3.1 e 3.3, o que indica que a maior parte dos candidatos possui desempenho acadêmico consistente. A mediana situa-se pouco acima de 3.2, reforçando esse padrão. Observa-se ainda a presença de alguns valores atípicos, tanto abaixo de 2.8 quanto acima de 3.6, que representam candidatos com desempenho fora do perfil predominante. Esses outliers, embora pouco frequentes, podem influenciar a análise estatística e devem ser considerados no pré-processamento ou na interpretação dos resultados do modelo. De forma geral, a distribuição do GPA sugere que a base é composta majoritariamente por candidatos academicamente fortes, o que pode ser um dos critérios determinantes no processo de admissão.

    ![Distribuição do GPA dos candidatos](docs/arvore-decisao/img/distribuicao_gpa_boxplot.png)

=== "major"
    A variável major, que representa a área de formação acadêmica dos candidatos, apresenta distribuição relativamente equilibrada entre as categorias, mas com destaque para Humanidades, que concentra o maior número de aplicações. As áreas de STEM e Business aparecem em proporções semelhantes, ambas com menor participação em relação a Humanidades. Essa diferença pode refletir o perfil da amostra, indicando maior procura de candidatos oriundos de cursos de Humanidades pelos programas de MBA. A análise dessa variável é relevante para verificar se determinadas formações acadêmicas têm maior representatividade ou desempenham papel diferenciado nos resultados de admissão.

    ![Frequência de áreas de formação (majors)](docs/arvore-decisao/img/frequencia_majors.png)

=== "race"
    A variável raça apresenta distribuição diversificada entre os candidatos, com destaque para a categoria de pessoas que preferiram não se identificar, seguida pelo grupo White. Em seguida aparecem Asian, Black e Hispanic, enquanto a categoria Other concentra a menor quantidade de aplicações. Essa composição evidencia tanto a representatividade de diferentes origens raciais quanto a limitação do campo para candidatos internacionais. A análise dessa variável é importante para compreender a diversidade do conjunto de dados e avaliar se há possíveis diferenças de perfil que podem influenciar nos resultados de admissão.

    ![Frequência de raça por aplicações](docs/arvore-decisao/img/frequencia_raca.png)    

=== "gmat"
    A variável GMAT apresenta uma distribuição concentrada entre 600 e 700 pontos, faixa onde se encontra a maior parte dos candidatos. O pico de frequência ocorre próximo de 650 pontos, o que sugere que esse valor é representativo do desempenho médio dos aplicantes. Apesar dessa concentração, também há candidatos com pontuações mais baixas, em torno de 570, bem como outros que alcançam notas elevadas acima de 750, embora em menor quantidade. Essa distribuição indica que, em geral, os candidatos possuem desempenho sólido no exame, mas com variação suficiente para permitir que o modelo identifique padrões relacionados ao status de admissão.

    ![Distribuição das pontuações de GMAT](docs/arvore-decisao/img/distribuicao_gmat.png)

=== "work_exp"
    A variável experiência profissional apresenta distribuição concentrada entre 4 e 6 anos de atuação no mercado, com destaque para os candidatos que possuem 5 anos de experiência, que representam a maior parte das aplicações. Os extremos da distribuição, com candidatos que possuem apenas 1 ou 2 anos de experiência e aqueles com mais de 7 anos, aparecem em menor número, configurando perfis menos frequentes na amostra. Esse padrão sugere que a base de dados está composta principalmente por profissionais em estágio intermediário de carreira, o que reflete o perfil típico de aplicantes a programas de MBA. Essa variável é particularmente relevante, pois pode influenciar diretamente nas chances de admissão, uma vez que a experiência prática é um critério valorizado nas seleções.

    ![Distribuição da experiência profissional](docs/arvore-decisao/img/distribuicao_experiencia_profissional.png)

=== "work_industry"
    A variável setor de experiência profissional revela que a maior parte dos candidatos possui trajetória em Consultoria, que se destaca amplamente em relação aos demais setores. Em seguida aparecem PE/VC (Private Equity e Venture Capital), Tecnologia e setores ligados ao serviço público ou organizações sem fins lucrativos, todos com participação significativa. Áreas tradicionais como Investment Banking e Financial Services também se mostram relevantes, mas em menor proporção. Já setores como Saúde, Bens de Consumo (CPG), Mídia/Entretenimento, Varejo, Imobiliário e Energia aparecem de forma mais restrita, representando nichos específicos da amostra. Essa distribuição indica que o MBA atrai predominantemente profissionais de consultoria e finanças, mas também apresenta diversidade ao incluir candidatos de áreas emergentes e de setores menos tradicionais.

    ![Frequência por setor de experiência profissional](docs/arvore-decisao/img/frequencia_setor_experiencia_profissional.png)

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
Após o pré-processamento, os dados foram divididos em dois subconjuntos: 80% para treinamento e 20% para teste. Essa proporção é adequada em problemas de classificação, pois garante exemplos suficientes para o aprendizado do modelo, ao mesmo tempo em que reserva uma amostra representativa para a avaliação final.

A divisão foi realizada utilizando a função train_test_split da biblioteca scikit-learn, com estratificação pela variável alvo (admission), de modo a preservar a proporção original entre as classes (Admit, Waitlist e Deny) em ambos os conjuntos. Isso assegura que o modelo não seja influenciado por distribuições desbalanceadas.

Dessa forma, o conjunto de treinamento é utilizado para armazenar os exemplos que servirão de referência ao algoritmo KNN, enquanto o conjunto de teste é aplicado para medir a acurácia e a capacidade de generalização do modelo na classificação de novos candidatos.

```python exec="0"
--8<-- "docs/arvore-decisao/divisao.py"
```
## Treinamento do Modelo

=== "Modelo"
    ```python exec="on" html="1"    
    --8<-- "docs/knn/modelo_knn.py"
    ```
=== "Código"
    ```python exec="0"    
    --8<-- "docs/knn/modelo_knn.py"
    ```

## Usando Scikit-Learn

=== "Resultado"
    ```python exec="on" html="1"    
    --8<-- "docs/knn/decision_knn.py"
    ```
=== "Código"
    ```python exec="0"    
    --8<-- "docs/knn/decision_knn.py"
    ```

## Avaliação do Modelo
O modelo KNN, configurado com k = 5, apresentou uma acurácia de 84% no conjunto de teste, evidenciando bom desempenho na tarefa de prever o status de admissão dos candidatos. Isso significa que, em média, oito a cada dez previsões realizadas foram corretas. A visualização da fronteira de decisão confirma esse resultado: a maior parte do espaço é dominada pela classe Deny, refletindo a predominância dessa categoria na base. Ainda assim, o modelo conseguiu identificar regiões específicas associadas às classes Waitlist e Admit, ainda que com certa sobreposição, o que explica eventuais erros de classificação.

O desempenho consistente do KNN reforça sua capacidade de capturar padrões de similaridade entre os candidatos, especialmente quando atributos como GPA e GMAT se combinam com variáveis de experiência profissional. Entretanto, a irregularidade da fronteira de decisão também evidencia uma limitação do método, que pode se tornar sensível a ruídos e depender fortemente da distribuição dos dados no espaço de features.

## Conclusão 
O uso do algoritmo KNN mostrou-se eficaz para a classificação dos candidatos ao MBA, atingindo bons níveis de acurácia e confirmando a relevância de atributos acadêmicos e profissionais na determinação do status de admissão. O modelo se destacou por sua simplicidade e pela interpretação intuitiva baseada em proximidade entre perfis semelhantes.

Apesar disso, a análise gráfica revelou que a separação entre classes não é totalmente nítida, em especial entre os grupos Admit e Waitlist, o que sugere a necessidade de ajustes no valor de k ou a utilização de técnicas complementares para melhorar a robustez das previsões. De maneira geral, os resultados demonstram que o KNN pode servir como uma ferramenta útil no apoio à análise de admissões, oferecendo uma abordagem alternativa e comparativa em relação à árvore de decisão, já aplicada anteriormente.