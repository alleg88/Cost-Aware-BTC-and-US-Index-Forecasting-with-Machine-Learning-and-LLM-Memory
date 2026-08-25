-- Q2 2026 GDELT GKG export used to complete the staged news corpus.
-- TranslationInfo is blank for documents originally published in English.
CREATE TEMP FUNCTION titleunescape(title STRING)
RETURNS STRING
LANGUAGE js AS """
  if (title === null) return null;
  return title.replace(/(&#x)([a-zA-Z0-9]+)(;)/gu, function(_, a, b) {
    return String.fromCodePoint(parseInt(b, 16));
  });
""";

WITH base AS (
  SELECT
    DATE,
    DocumentIdentifier AS url,
    NET.REG_DOMAIN(DocumentIdentifier) AS domain,
    V2Tone,
    titleunescape(
      REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)<\/PAGE_TITLE>')
    ) AS title,
    LOWER(CONCAT(
      IFNULL(DocumentIdentifier, ''), ' ',
      IFNULL(Extras, ''), ' ',
      IFNULL(V2Themes, ''), ' ',
      IFNULL(AllNames, '')
    )) AS topic_text
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE _PARTITIONTIME >= TIMESTAMP('2026-04-01')
    AND _PARTITIONTIME < TIMESTAMP('2026-07-01')
    AND COALESCE(TranslationInfo, '') = ''
    AND NET.REG_DOMAIN(DocumentIdentifier) IN (
      'apnews.com',
      'barrons.com',
      'bitcoinmagazine.com',
      'bloomberg.com',
      'businessinsider.com',
      'cnbc.com',
      'coindesk.com',
      'cointelegraph.com',
      'decrypt.co',
      'forbes.com',
      'fortune.com',
      'ft.com',
      'investing.com',
      'marketwatch.com',
      'reuters.com',
      'theblock.co',
      'wsj.com'
    )
    AND REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)<\/PAGE_TITLE>') IS NOT NULL
),
classified AS (
  SELECT
    DATE,
    url,
    domain,
    V2Tone,
    title,
    CASE
      WHEN REGEXP_CONTAINS(
        topic_text,
        r'(bitcoin|\bbtc\b|cryptocurrency)'
      ) THEN 'btc'
      WHEN REGEXP_CONTAINS(
        LOWER(CONCAT(IFNULL(title, ''), ' ', IFNULL(url, ''))),
        r'(nasdaq|tech stocks?|big tech)'
      ) THEN 'usatech'
      WHEN REGEXP_CONTAINS(
        topic_text,
        r'(s&p 500|s&amp;p 500|s%26p[-_ ]?500|\bsp500\b|wall street|federal reserve|\bcpi\b|jobs report|nasdaq|tech stocks?|big tech)'
      ) THEN 'usa500'
      ELSE NULL
    END AS stream
  FROM base
)
SELECT DATE, url, domain, V2Tone, title, stream
FROM classified
WHERE stream IS NOT NULL
ORDER BY DATE, url;
