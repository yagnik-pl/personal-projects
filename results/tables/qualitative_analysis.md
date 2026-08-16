# Qualitative Query Analysis by Exit Layer

Configuration: `threshold=0.85`, `min_layer=10`

## Layer Group Statistics

|   exit_layer |   num_queries |   pct_queries |   avg_word_count |   avg_char_count |   hit@1_rate |   hit@10_rate |   avg_stability |
|-------------:|--------------:|--------------:|-----------------:|-----------------:|-------------:|--------------:|----------------:|
|           10 |           216 |          72   |             12.7 |             91.9 |       0.0417 |        0.1574 |          0.869  |
|           11 |            26 |           8.7 |             11.2 |             81.7 |       0.5385 |        0.8462 |          0.858  |
|           12 |            58 |          19.3 |             12.2 |             88.3 |       0.6897 |        0.8103 |          0.9003 |

## Representative Queries

### Exit Layer 10 (Sample of 3 queries)

- **Query**: "0-dimensional biomaterials show inductive properties."
  - Length: 5 words | Exit Stability: 0.8872 | Hit@10: False
- **Query**: "1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants."
  - Length: 20 words | Exit Stability: 0.8511 | Hit@10: True
- **Query**: "1/2000 in UK have abnormal PrP positivity."
  - Length: 7 words | Exit Stability: 0.8764 | Hit@10: False

### Exit Layer 11 (Sample of 3 queries)

- **Query**: "Asymptomatic visual impairment screening in elderly populations does not lead to improved vision."
  - Length: 13 words | Exit Stability: 0.8671 | Hit@10: True
- **Query**: "Bariatric surgery has a positive impact on mental health."
  - Length: 9 words | Exit Stability: 0.8612 | Hit@10: True
- **Query**: "CX3CR1 on the Th2 cells impairs T cell survival"
  - Length: 9 words | Exit Stability: 0.8584 | Hit@10: True

### Exit Layer 12 (Sample of 3 queries)

- **Query**: "A total of 1,000 people in the UK are asymptomatic carriers of vCJD infection."
  - Length: 14 words | Exit Stability: 0.8937 | Hit@10: True
- **Query**: "Anthrax spores can be disposed of easily after they are dispersed."
  - Length: 11 words | Exit Stability: 0.9181 | Hit@10: True
- **Query**: "Antiretroviral therapy reduces rates of tuberculosis across a broad range of CD4 strata."
  - Length: 13 words | Exit Stability: 0.8835 | Hit@10: True

