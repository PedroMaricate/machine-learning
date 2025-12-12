## Introdução
O modelo desenvolvido tem como objetivo prever o resultado de admissões em programas de MBA a partir de um conjunto de informações sobre os candidatos. Para isso, foi escolhida a técnica de árvores de decisão, que se mostra adequada por sua interpretabilidade e pela capacidade de lidar com variáveis de diferentes naturezas. Essa abordagem permite não apenas realizar previsões sobre o status de admissão, mas também compreender quais fatores exercem maior influência no processo seletivo. Dessa forma, o modelo não se limita a um exercício de classificação, mas também funciona como ferramenta de análise, apontando as características mais relevantes no perfil de um candidato admitido, em lista de espera ou recusado.

## Base de dados
A [base](https://www.kaggle.com/datasets/taweilo/mba-admission-dataset) utilizada é composta por dados sintéticos criados a partir das estatísticas da turma de 2025 do MBA de Wharton. Ela reúne informações demográficas, acadêmicas e profissionais de candidatos, como gênero, nacionalidade, área de formação, desempenho no GPA e no GMAT, além de experiência de trabalho e setor de atuação. Esses atributos foram relacionados ao status final da candidatura, categorizado como admitido, em lista de espera ou negado. Por se tratar de um conjunto de dados diversificado, é possível observar tanto os aspectos objetivos ligados ao desempenho acadêmico e profissional quanto elementos contextuais que podem influenciar o resultado do processo seletivo. Essa combinação torna o dataset especialmente relevante para análises exploratórias e para o desenvolvimento de modelos preditivos que buscam compreender os critérios implícitos de seleção em admissões de MBA.



## Exploração dos Dados

A seguir foi realizada uma análise exploratória da base de dados, com o objetivo de compreender o significado e a composição de cada coluna. Essa etapa busca identificar possíveis problemas, como valores ausentes ou distribuições desbalanceadas, que podem influenciar diretamente a qualidade do modelo. As visualizações e estatísticas descritivas permitem observar padrões, tendências e discrepâncias entre os candidatos, fornecendo subsídios importantes para orientar as decisões de pré-processamento e a construção da árvore de decisão.

=== "gender"
    A variável gênero apresenta uma diferença significativa na quantidade de aplicações. Observa-se uma predominância de candidatos do sexo masculino em comparação às candidatas do sexo feminino, o que indica uma distribuição desigual nesse aspecto. Essa discrepância pode refletir tendências do mercado de MBA ou características específicas da base gerada. Além disso, é um fator importante a ser considerado no modelo, já que possíveis vieses de gênero podem influenciar tanto a análise quanto as previsões de admissão.

    ![Frequência de gênero por aplicações](frequencia_genero_aplicacoes.png)


=== "international"
    A variável alunos internacionais mostra que a maior parte das aplicações é de candidatos domésticos (não internacionais), enquanto os estudantes internacionais representam uma parcela menor do total. Essa diferença pode indicar que os programas de MBA ainda têm maior procura local, embora o número de aplicações internacionais seja relevante para demonstrar a diversidade e a atratividade global da instituição. Essa característica pode influenciar o modelo de predição, visto que fatores como origem do aluno podem estar relacionados às taxas de aceitação.

    ![Frequência de alunos internacionais por aplicações](frequencia_alunos_internacionais.png)


=== "gpa"
    A variável GPA apresenta distribuição concentrada em torno de valores relativamente altos, entre 3.1 e 3.3, o que indica que a maior parte dos candidatos possui desempenho acadêmico consistente. A mediana situa-se pouco acima de 3.2, reforçando esse padrão. Observa-se ainda a presença de alguns valores atípicos, tanto abaixo de 2.8 quanto acima de 3.6, que representam candidatos com desempenho fora do perfil predominante. Esses outliers, embora pouco frequentes, podem influenciar a análise estatística e devem ser considerados no pré-processamento ou na interpretação dos resultados do modelo. De forma geral, a distribuição do GPA sugere que a base é composta majoritariamente por candidatos academicamente fortes, o que pode ser um dos critérios determinantes no processo de admissão.

    ![Distribuição do GPA dos candidatos](distribuicao_gpa_boxplot.png)

=== "major"
    A variável major, que representa a área de formação acadêmica dos candidatos, apresenta distribuição relativamente equilibrada entre as categorias, mas com destaque para Humanidades, que concentra o maior número de aplicações. As áreas de STEM e Business aparecem em proporções semelhantes, ambas com menor participação em relação a Humanidades. Essa diferença pode refletir o perfil da amostra, indicando maior procura de candidatos oriundos de cursos de Humanidades pelos programas de MBA. A análise dessa variável é relevante para verificar se determinadas formações acadêmicas têm maior representatividade ou desempenham papel diferenciado nos resultados de admissão.

    ![Frequência de áreas de formação (majors)](frequencia_majors.png)


=== "race"
    A variável raça apresenta distribuição diversificada entre os candidatos, com destaque para a categoria de pessoas que preferiram não se identificar, seguida pelo grupo White. Em seguida aparecem Asian, Black e Hispanic, enquanto a categoria Other concentra a menor quantidade de aplicações. Essa composição evidencia tanto a representatividade de diferentes origens raciais quanto a limitação do campo para candidatos internacionais. A análise dessa variável é importante para compreender a diversidade do conjunto de dados e avaliar se há possíveis diferenças de perfil que podem influenciar nos resultados de admissão.

    ![Frequência de raça por aplicações](frequencia_raca.png)
   

=== "gmat"
    A variável GMAT apresenta uma distribuição concentrada entre 600 e 700 pontos, faixa onde se encontra a maior parte dos candidatos. O pico de frequência ocorre próximo de 650 pontos, o que sugere que esse valor é representativo do desempenho médio dos aplicantes. Apesar dessa concentração, também há candidatos com pontuações mais baixas, em torno de 570, bem como outros que alcançam notas elevadas acima de 750, embora em menor quantidade. Essa distribuição indica que, em geral, os candidatos possuem desempenho sólido no exame, mas com variação suficiente para permitir que o modelo identifique padrões relacionados ao status de admissão.

    ![Distribuição das pontuações de GMAT](distribuicao_gmat.png)


=== "work_exp"
    A variável experiência profissional apresenta distribuição concentrada entre 4 e 6 anos de atuação no mercado, com destaque para os candidatos que possuem 5 anos de experiência, que representam a maior parte das aplicações. Os extremos da distribuição, com candidatos que possuem apenas 1 ou 2 anos de experiência e aqueles com mais de 7 anos, aparecem em menor número, configurando perfis menos frequentes na amostra. Esse padrão sugere que a base de dados está composta principalmente por profissionais em estágio intermediário de carreira, o que reflete o perfil típico de aplicantes a programas de MBA. Essa variável é particularmente relevante, pois pode influenciar diretamente nas chances de admissão, uma vez que a experiência prática é um critério valorizado nas seleções.

    ![Distribuição da experiência profissional](distribuicao_experiencia_profissional.png)


=== "work_industry"
    A variável setor de experiência profissional revela que a maior parte dos candidatos possui trajetória em Consultoria, que se destaca amplamente em relação aos demais setores. Em seguida aparecem PE/VC (Private Equity e Venture Capital), Tecnologia e setores ligados ao serviço público ou organizações sem fins lucrativos, todos com participação significativa. Áreas tradicionais como Investment Banking e Financial Services também se mostram relevantes, mas em menor proporção. Já setores como Saúde, Bens de Consumo (CPG), Mídia/Entretenimento, Varejo, Imobiliário e Energia aparecem de forma mais restrita, representando nichos específicos da amostra. Essa distribuição indica que o MBA atrai predominantemente profissionais de consultoria e finanças, mas também apresenta diversidade ao incluir candidatos de áreas emergentes e de setores menos tradicionais.

    ![Frequência por setor de experiência profissional](frequencia_setor_experiencia_profissional.png)

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
Após o pré-processamento, os dados foram divididos em dois subconjuntos: 80% para treinamento e 20% para teste. Essa proporção é amplamente utilizada em problemas de classificação, pois fornece uma quantidade suficiente de exemplos para o aprendizado do modelo, ao mesmo tempo em que reserva uma amostra representativa para a avaliação final.

A divisão foi realizada utilizando a função train_test_split da biblioteca scikit-learn, com a opção de estratificação pela variável alvo (admission), de modo a preservar a proporção original entre as classes (Admit, Waitlist e Deny) em ambos os conjuntos. Isso garante que o modelo não seja favorecido ou prejudicado por distribuições desbalanceadas.

Dessa forma, o conjunto de treinamento é usado para ajustar os parâmetros da árvore de decisão, enquanto o conjunto de teste serve como base imparcial para medir a acurácia e capacidade de generalização do modelo.

```python exec="0"
--8<-- "docs/arvore-decisao/divisao.py"
```

## Treinamento do Modelo

![Árvore](decision_tree.png)

=== "Código"
    ```python exec="0"    
    --8<-- "docs/arvore-decisao/decision-tree.py"
    ```

## Avaliação do modelo

O modelo de árvore de decisão alcançou uma acurácia de 78% na base de teste, demonstrando um desempenho satisfatório na tarefa de prever o status de admissão dos candidatos. Esse resultado indica que, em média, cerca de oito em cada dez previsões realizadas pelo modelo estão corretas, o que sugere boa capacidade de generalização.

A análise da importância das variáveis reforça a relevância dos indicadores acadêmicos, em especial o GPA (0.31) e o GMAT (0.30), que juntos representam mais de 60% da capacidade explicativa do modelo. Em seguida, fatores ligados à trajetória profissional, como setor de atuação (work_industry) e anos de experiência (work_exp), também apresentam peso significativo, ainda que menor, o que evidencia que o desempenho acadêmico continua sendo o principal critério de admissão.

As demais variáveis, como major, raça, nacionalidade e gênero, apresentam importâncias mais baixas, sugerindo que exercem influência menos direta sobre a decisão de admissão. No entanto, sua inclusão no modelo contribui para capturar nuances adicionais que podem afetar o resultado em casos específicos.

De forma geral, os resultados confirmam que a combinação de desempenho acadêmico e experiência profissional são os fatores mais determinantes para o processo de admissão, em consonância com práticas comuns em programas de MBA de alto nível.

## Conclusão
A partir da análise da base de dados de admissões em MBA, foi possível compreender o perfil dos candidatos e identificar padrões relevantes que influenciam os resultados de aceitação, lista de espera ou recusa. A exploração inicial revelou um conjunto diversificado de informações, abrangendo aspectos acadêmicos, profissionais e demográficos, permitindo visualizar tanto tendências predominantes — como a maior representatividade de determinados grupos — quanto a presença de outliers em variáveis como GPA e GMAT.

O pré-processamento dos dados foi fundamental para garantir consistência e qualidade, contemplando a padronização de variáveis categóricas, a imputação de valores ausentes e a definição clara entre variáveis explicativas e alvo. Essa preparação assegurou que o modelo pudesse ser treinado de forma adequada, sem vieses estruturais oriundos da própria base.

Na etapa de modelagem, a divisão entre treino e teste possibilitou avaliar a capacidade de generalização do algoritmo. O modelo de árvore de decisão alcançou uma acurácia de 78%, demonstrando desempenho sólido ao prever o status de admissão. A análise das importâncias das variáveis evidenciou a centralidade dos critérios acadêmicos — com destaque para GPA e GMAT — complementados por fatores ligados à experiência profissional, como anos de atuação e setor de trabalho. Aspectos demográficos, como gênero, raça e nacionalidade, apresentaram impacto secundário, contribuindo de forma menos expressiva para a classificação final.

De maneira geral, os resultados confirmam que o processo de admissão em programas de MBA é fortemente guiado pela combinação de desempenho acadêmico e experiência profissional, refletindo a prática de selecionar candidatos que conciliem excelência acadêmica com vivência de mercado. O estudo mostra, portanto, que modelos de aprendizado de máquina podem não apenas auxiliar na previsão de admissões, mas também oferecer insights relevantes sobre os fatores que mais pesam em processos seletivos competitivos.

